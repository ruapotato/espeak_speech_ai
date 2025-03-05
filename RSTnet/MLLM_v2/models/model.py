# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial
import logging
import typing as tp
import torch.nn.functional as F
import torch
from torch import nn

from utils.sampling import sample_token
from utils.compile import CUDAGraphed
from modules.streaming import StreamingContainer, StreamingModule
from modules.transformer import (
    StreamingTransformer,
    create_norm_fn,
)
from einops import rearrange


logger = logging.getLogger(__name__)

def CrossEntropyAndAccuracy(logits, y, masks, loss_weights, ignore_ids=None):
    """whether to use ignore_id=0
       we should calculate the loss for different layers, and set different weight for each layers
       mask: B, N, T
    """
    y = y.to(logits.device)
    masks = masks.to(logits.device)
    loss = 0
    num_all_tokens = 0
    taret_all_token = 0
    acc_tk  = 0
    target_acc = 0
    for idx, w in enumerate(loss_weights):
        # B, T, 8, card
        tmp_mask = masks[:,idx,:].reshape(-1) # 
        tmp_logit = logits[:,:,idx,:].reshape(-1, logits.shape[-1]).contiguous()
        tmp_y = y[:,idx,:].reshape(-1).contiguous()
        tmp_loss = F.cross_entropy(tmp_logit, tmp_y, ignore_index=ignore_ids[idx], reduction='none')
        tmp_loss = tmp_loss*tmp_mask # add mask
        tmp_pred = tmp_logit.argmax(1) # 
        tmp_num_all_tokens = tmp_mask.ne(0.0).int().sum() # we only calculate the non-mask part
        tmp_target_all_tokens = tmp_mask.eq(1.0).int().sum() # 

        tmp_acc_tk = torch.logical_and(tmp_pred.eq(tmp_y), tmp_mask.ne(0.0)).int().sum()
        tmp_acc_target = torch.logical_and(tmp_pred.eq(tmp_y), tmp_mask.eq(1.0)).int().sum()
        
        num_all_tokens += tmp_num_all_tokens
        acc_tk += tmp_acc_tk
        target_acc += tmp_acc_target
        taret_all_token += tmp_target_all_tokens
        loss += (tmp_loss.sum()/tmp_num_all_tokens)*loss_weights[idx]
    acc_all = acc_tk/num_all_tokens
    acc_target = target_acc/taret_all_token
    metrics = {'acc_all': acc_all, 'acc_target': acc_target, 'loss': loss.clone().detach()}
    return loss, metrics

class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).

    Args:
        norm (bool): if True, uses a layer norm after the embedding.
        zero_idx (int): special value indicating that the output should be exactly 0.
    """

    def __init__(self, *args, norm: bool = False, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = None
        if norm:
            self.norm = create_norm_fn("layer_norm", self.embedding_dim)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        zero = torch.zeros(1, dtype=input.dtype, device=input.device)
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        if self.norm is not None:
            y = self.norm(y)
        y = torch.where(is_zero[..., None], zero, y)
        return y

class ModelConfig:
    def __init__(self, model_type):
        self.model_type = model_type
    

class LMModel(StreamingContainer):
    """Transformer-based language model on multiple streams of codes.

    Args:
        n_q (int): Number of parallel streams to model as input.
        dep_q (int): Number of parallel streams to model in the depformer.
        card (int): Cardinality, vocabulary size.
        text_card (int): Cardinality of the text vocabulary.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_emb (bool): Whether to normalize embeddings.
        bias_proj (bool): Use bias for output projections.
        depformer_*: params used for the Depformer Transformer, all the other will be shared.
        depformer_multi_linear (bool): if True, uses one linear layer per codebook to project the
            output of the main transformer to the Depformer latent space.
        depformer_dim_feedforward (int| list[int]| None): If None, defaults to hidden_scale * depformer_dim.
        existing_text_padding_id (bool): if True, will use a different token for the initial text token, and
            the text padding token.
        same_initial (bool): if True, uses the same initial tokens for both text and audio mode.
        **kwargs: Additional parameters for the transformer encoder.
    """

    def __init__(
        self,
        delays: tp.List[int] = [0],
        n_q: int = 8,
        dep_q: int = 8,
        card: int = 1024,
        text_card: int = 32000,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_emb: bool = False,
        bias_proj: bool = False,
        depformer_dim: int = 256,
        depformer_dim_feedforward: int | list[int] | None = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_pos_emb: str = "sin",
        existing_text_padding_id: tp.Optional[int] = None,
        context: tp.Optional[int] = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.n_q = n_q
        self.dep_q = dep_q
        self.card = card
        self.text_card = text_card
        assert len(delays) == self.num_codebooks, "unexpected number of delays"
        self.delays = delays
        self.dim = dim
        self.existing_text_padding_id = existing_text_padding_id
        self.context = context
        kwargs["context"] = context
        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.zero_token_id,
        )
        self.emb = nn.ModuleList([EmbeddingFactory(self.card + 1, dim) for _ in range(n_q)]) # build embedding mapping for each codebook
        # Text card + padding token (if not in the original tokenizer)
        extra_text = self.existing_text_padding_id is None # if the original tokenizer,such as sentencepiece does not include a padding token
        # Unlike for audio, here we authorize the model to output the special token.
        self.text_emb = EmbeddingFactory(text_card + 1, dim) # for text
        self.text_linear = nn.Linear(dim, text_card + extra_text, bias=bias_proj) # output space
        depformer_prefix = "depformer_"
        main_kwargs = {k: v for k, v in kwargs.items() if not k.startswith(depformer_prefix)}
        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm,
            device=device,
            dtype=dtype,
            **main_kwargs,
        )
        self.out_norm = create_norm_fn(norm, dim)
        self.depformer_multi_linear = depformer_multi_linear
        kwargs_dep = main_kwargs.copy()
        kwargs_dep.update(
            {
                k.removeprefix(depformer_prefix): v
                for k, v in kwargs.items()
                if k.startswith(depformer_prefix)
            }
        )
        kwargs_dep["positional_embedding"] = depformer_pos_emb
        kwargs_dep["context"] = None
        if depformer_weights_per_step:
            kwargs_dep["weights_per_step"] = dep_q
        if depformer_multi_linear: #
            # One linear layer per codebook to project different informations from the main model.
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False) for _ in range(dep_q)]
            ) # for the depth transformer, we use different linear layer to map features dep_q=8. We only output our stream
        else:
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False)]
            )
        # Only using up to dep_q - 1 because the last codebook is never an input to Depformer.
        self.depformer_emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, depformer_dim) for _ in range(dep_q - 1)]
        )
        self.depformer_text_emb = EmbeddingFactory(text_card + 1, depformer_dim)
        if depformer_dim_feedforward is None:
            depformer_dim_feedforward = int(hidden_scale * depformer_dim)
        self.depformer = StreamingTransformer(
            d_model=depformer_dim,
            dim_feedforward=depformer_dim_feedforward,
            norm=norm,
            device=device,
            dtype=dtype,
            **kwargs_dep,
        )
        self.depformer.set_streaming_propagate(False)
        dim = depformer_dim  # we will directly apply the next linears to the output of the Depformer.

        self.linears = nn.ModuleList(
            [nn.Linear(dim, self.card, bias=bias_proj) for _ in range(dep_q)]
        ) # output for each layer
        self.config = ModelConfig(model_type='lora')
    @property
    def initial_token_id(self) -> int:
        """Token id for the start of sequence (audio)."""
        return self.card

    @property
    def text_initial_token_id(self) -> int:
        """Token id for the start of sequence (text)."""
        return self.text_card

    @property
    def text_padding_token_id(self) -> int:
        """Token id for text padding."""
        if self.existing_text_padding_id is None:
            return self.text_card
        else:
            return self.existing_text_padding_id

    @property
    def end_of_text_padding_id(self) -> int:
        """Token id for optionally marking the last padding step for a word."""
        return 0

    @property
    def zero_token_id(self) -> int:
        """Special value in the input tokens, indicating that no sampling should
        happen for that value, and no input should be given to the model."""
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        """Special value that can be provided in the prompt to indicate that this specific
        value should be predicted and sampled. This allows for partial teacher forcing, by generating
        one modality, with the other one fixed.
        """
        return -2

    @property
    def device(self):
        first_param = next(iter(self.parameters()))
        return first_param.device

    @property
    def num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    @property
    def audio_offset(self) -> int:
        return 1

    def _get_initial_token(self) -> torch.Tensor:
        # Returns the initial token that will be fed to the model to predict the very first timestep. 初始化1帧 (1,17,1)
        # The output shape will be [B, K, 1].
        device = next(iter(self.parameters())).device
        zero = torch.full(
            [1, 1, 1], self.zero_token_id, device=device, dtype=torch.long
        )
        special = torch.full_like(zero, self.initial_token_id)

        text_special = torch.full_like(zero, self.text_initial_token_id)
        audio_token = special
        text_token = text_special
        audio_token = audio_token.expand(-1, self.num_audio_codebooks, -1)
        token = torch.cat([text_token, audio_token], dim=1)
        return token


    def forward(self, sequence: torch.Tensor, masks: torch.Tensor = None):
        """ training forward function
        global forward: padding a frame. In fact, for casual attention, we donot need mask for attention. We only use it to avoid loss calculation
        """
        B, K, S = sequence.shape  # 
        #print('sequence ', sequence.shape)
        global_start_frame = self._get_initial_token() # get the 
        global_start_frame = global_start_frame.repeat(B,1,1)
        #print('global_start_frame ', global_start_frame.shape)
        global_input_sequence = torch.cat([global_start_frame, sequence[:,:,:-1]], dim=2) # add start token
        transformer_out, text_logits = self.forward_text(global_input_sequence)
        text_logits = text_logits.squeeze(1) # B, T, D
        #text_indices = sequence[:,0,:] # B, T for the training stage, we directly use the gt
        text_indices = global_input_sequence[:,0,:] # B, T for the training stage, we directly use the gt
        #print('text_indices ', text_indices.shape)
        local_start_token = self.depformer_text_emb(text_indices) # using text embedding for local start token, B,T,D
        # print('local_start_token ', local_start_token.shape)
        #local_sequence = sequence[:,1:self.dep_q+1,:] # B, 8, T
        local_sequence = global_input_sequence[:,1:self.dep_q+1,:] # the input for local sequence
        audio_logits = self.forward_local(local_start_token, local_sequence, transformer_out) # B, T, 8, card
        # print('audio_logits ', audio_logits.shape)
        # print('text_logits ', text_logits.shape)
        return audio_logits, text_logits

    def forward_local(
        self,
        local_start_token: torch.Tensor,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        B, K, S = sequence.shape
        assert (K == self.dep_q), f"Sequence shape {sequence.shape} must match the moshi stream output."
        depformer_input = transformer_out
        local_inputs = []
        local_start_token = local_start_token.reshape(-1, local_start_token.shape[-1]) # transfer to 
        #print('local_start_token ', local_start_token.shape)
        local_inputs.append(local_start_token)
        different_view_depformer = []
        for cb_index in range(self.dep_q-1): # 7
            local_token_input = self.depformer_emb[cb_index](sequence[:,cb_index:cb_index+1,:]) # get the local embedding, B, 1, T,D
            local_token_input = local_token_input.reshape(-1, local_token_input.shape[-1])
            #print('local_token_input ', local_token_input.shape)
            local_inputs.append(local_token_input) # B*T,D

        for cb_index in range(self.dep_q):
            tmp_dep_input = self.depformer_in[cb_index](depformer_input)
            #print('tmp_dep_input ', tmp_dep_input.shape)
            tmp_dep_input = tmp_dep_input.reshape(-1, tmp_dep_input.shape[-1])
            #print('tmp_dep_input ', tmp_dep_input.shape)
            different_view_depformer.append((tmp_dep_input+local_inputs[cb_index]).unsqueeze(1)) #B*T,1, D

        real_depformer_input = torch.cat(different_view_depformer, dim=1) # B*T, 8, D
        #print('real_depformer_input ', real_depformer_input.shape)
        # depformer_input is [B, 1, depformer_dim].
        # The streaming state of the depformer ensures that the proper layer is run.
        dep_output = self.depformer(real_depformer_input) # B*T, 8, D
        #print('dep_output ', dep_output.shape)
        logits = []
        for depformer_cb_index in range(self.dep_q):
            tmp_logit = self.linears[depformer_cb_index](dep_output[:,depformer_cb_index:depformer_cb_index+1,:]) # B*T,1,card
            tmp_logit = tmp_logit.reshape(B, -1, 1, tmp_logit.shape[-1]) # B, T, 1, card
            #print('tmp_logit ', tmp_logit.shape)
            logits.append(tmp_logit)
        logits = torch.cat(logits, dim=2)  # B, T, 8, card
        assert logits.dim() == 4, logits.shape  # ?
        return logits

    def forward_text(
        self,
        sequence: torch.Tensor, masks: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, S = sequence.shape
        assert (
            K == self.num_codebooks
        ), f"Sequence shape {sequence.shape} must match the number of codebooks."
        input_sequence = sequence
        input_ = None
        for cb_index in range(self.num_audio_codebooks): # 16
            audio_emb = self.emb[cb_index](
                input_sequence[:, cb_index + self.audio_offset] # start from 1
            ) # different codebook use different layer
            input_ = audio_emb if input_ is None else input_ + audio_emb # using add operation to merge all of the information
        text_emb = self.text_emb(input_sequence[:, 0]) # encode the text information
        input_ = text_emb if input_ is None else input_ + text_emb # similarly, add the information
        #print('input_ e ', input_.shape)
        transformer_out = self.transformer(input_)
        #print('transformer_out ', transformer_out.shape)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)
        text_logits = self.text_linear(transformer_out) # the transformer_out first be used as the hidden state for text prediction
        text_logits = text_logits[:, None]
        return transformer_out, text_logits


    def forward_depformer(
        self,
        depformer_cb_index: int,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        B, K, S = sequence.shape
        assert (
            K == 1
        ), f"Codebooks for Depformer streaming should be passed 1 by 1, got {K}."
        assert (
            S == 1
        ), f"Steps for Depformer streaming should be passed 1 by 1, got {S}."
        assert (
            transformer_out.shape[1] == 1
        ), "Transformer out should be a for a single step."
        last_token_input: tp.Optional[torch.Tensor] = None
        depformer_input = transformer_out
        if self.depformer_multi_linear:
            depformer_input = self.depformer_in[depformer_cb_index](depformer_input) # 
        else:
            depformer_input = self.depformer_in[0](depformer_input)
        if depformer_cb_index == 0: 
            last_token_input = self.depformer_text_emb(sequence[:, 0]) # use the text_emb?
        else:
            last_token_input = self.depformer_emb[depformer_cb_index - 1](
                sequence[:, 0]
            )
        depformer_input = depformer_input + last_token_input # add previous embedding?
        assert depformer_input.shape[1] == 1
        # depformer_input is [B, 1, depformer_dim].
        # The streaming state of the depformer ensures that the proper layer is run.
        dep_output = self.depformer(depformer_input)
        logits = self.linears[depformer_cb_index](dep_output)
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, Ka, S, card]
        return logits


@dataclass
class _LMGenState:
    cache: torch.Tensor
    initial: torch.Tensor
    graphed_main: CUDAGraphed
    graphed_depth: CUDAGraphed
    offset: int = 0

    def reset(self):
        self.offset = 0


class LMGen(StreamingModule[_LMGenState]):
    def __init__(
        self,
        lm_model: LMModel,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        check: bool = False,
    ):
        assert not lm_model.training, "generation shouldn't be used in training mode."
        super().__init__()

        self.lm_model = lm_model
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.check = check
        self.max_delay = max(
            lm_model.delays
        )  # with delays, we need to generate a few more time steps.
        self.delays_cuda = torch.tensor(
            lm_model.delays, device=lm_model.device, dtype=torch.long
        )
     
    def _init_streaming_state(self, batch_size: int) -> _LMGenState:
        """ this function will be inited after start
        """
        lm_model = self.lm_model
        initial = lm_model._get_initial_token() # 1, 17, 1
        # print('batch_size ', batch_size, self.lm_model.num_codebooks, self.max_delay)
        # assert 1==2
        cache = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 2),
            lm_model.ungenerated_token_id,
            device=lm_model.device,
            dtype=torch.long,
        ) # self.max_delay: 1, ungenerated_token_id: -2
        disable = lm_model.device.type != 'cuda'
        graphed_main = CUDAGraphed(lm_model.forward_text, disable=disable) # forward_text
        graphed_depth = CUDAGraphed(self.depformer_step, disable=disable) # depformer_step

        return _LMGenState(cache, initial, graphed_main, graphed_depth)

    @torch.no_grad()
    def step(self, input_tokens: torch.Tensor) -> torch.Tensor | None:
        state = self._streaming_state
        if state is None:
            raise RuntimeError(
                "You should wrap those calls with a `with lm_gen.streaming(): ...`."
            )
        lm_model = self.lm_model

        assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
        B, Ki, S = input_tokens.shape
        assert S == 1, "Only support being given steps one by one."
        needed_tokens = lm_model.num_codebooks - lm_model.dep_q - 1 # 17-8-1
        assert (
            Ki == needed_tokens
        ), f"We expect {needed_tokens} tokens from the user stream, got {Ki}."
        CT = state.cache.shape[2] # 3?
        for q_other in range(input_tokens.shape[1]): # pass the user input data
            k = lm_model.dep_q + 1 + q_other # q_other: 0,1,2,3,4,5,6,7
            #print('k ', k) # k: 9,....16
            delay = lm_model.delays[k] # [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], get the corresponding delay value for user stream
            write_position = (state.offset + delay) % CT
            #print('write_position ', write_position)
            state.cache[:, k, write_position : write_position + 1] = input_tokens[
                :, q_other
            ]
        position = state.offset % CT # 0
        for k, delay in enumerate(lm_model.delays):
            # Only for the very beginning, we extend the initial token for the acoustic
            # token that are delayed, and thus have no good value to take.
            if state.offset <= delay:
                state.cache[:, k, position] = state.initial[:, k, 0]
        input_ = state.cache[:, :, position : position + 1]
        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                state.offset,
                input_,
            )
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()
        transformer_out, text_logits = state.graphed_main(input_) # input the first frame
        # Shape of text_logits should be [B, K_text=1, T=1, Card_text]
        text_token = sample_token(
            text_logits.float(),
            self.use_sampling,
            self.temp_text,
            self.top_k_text,
        )
        assert text_token.dim() == 3, text_token.shape
        assert text_token.shape[2] == 1
        assert text_token.shape[1] == 1, "Only one text stream supported."
        text_token = text_token[:, 0, 0]  # shape is [B]
        audio_tokens = state.graphed_depth(text_token, transformer_out)
        # ensure we don't overwrite prompt tokens, we only write over ungenerated tokens
        state.offset += 1
        position = state.offset % CT
        state.cache[:, 0, position] = text_token
        state.cache[:, 1 : lm_model.dep_q + 1, position] = audio_tokens # move to the next cache
        # print('state.offset ', state.offset, self.max_delay)
        # assert 1==2
        if state.offset <= self.max_delay:
            print('none')
            return None
        B = state.cache.shape[0]
        gen_delays_cuda = self.delays_cuda[: lm_model.dep_q + 1] # 9
        index = (
            ((state.offset - self.max_delay + gen_delays_cuda) % CT)
            .view(1, -1, 1)
            .expand(B, -1, 1)
        )
        out = state.cache.gather(dim=2, index=index)
        return out

    def depformer_step(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        (B,) = text_token.shape
        prev_token = text_token # text token is out
        lm_model = self.lm_model
        depformer_tokens: list[torch.Tensor] = []
        assert not lm_model.depformer.is_streaming
        # print('lm_model.dep_q ', lm_model.dep_q) # 8
        # assert 1==2
        with lm_model.depformer.streaming(B): # 可以确保在处理 depformer 时，任何相关的状态和流处理逻辑都是独立的
            for cb_index in range(lm_model.dep_q):
                input_ = prev_token[:, None, None] # 
                logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
                next_token = sample_token(
                    logits.float(),
                    self.use_sampling,
                    self.temp,
                    self.top_k,
                )
                assert next_token.shape == (B, 1, 1)
                next_token = next_token[:, 0, 0]  # shape is B
                depformer_tokens.append(next_token)
                prev_token = next_token

        assert len(depformer_tokens) == lm_model.dep_q, (
            len(depformer_tokens),
            lm_model.dep_q,
        )
        out = torch.stack(depformer_tokens, dim=1)
        assert out.shape == (B, lm_model.dep_q), out.shape
        return out
