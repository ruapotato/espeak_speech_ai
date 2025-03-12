from datasets import load_dataset
import json
import os
import re

def fix_dataset(dataset):
    """
    Reconstructs proper conversations from the dataset based on the assumption
    that the dataset contains alternating user and assistant messages, but they're
    not properly labeled.
    """
    train_data = dataset['train']
    reconstructed_convs = []
    
    # Group by conversation (assuming linear conversation order)
    current_conv = []
    
    for i in range(len(train_data)):
        msg = train_data[i]
        
        # Start a new conversation every N items
        if i % 10 == 0 and current_conv:
            reconstructed_convs.append(current_conv)
            current_conv = []
        
        current_conv.append(msg)
    
    # Add the last conversation if it exists
    if current_conv:
        reconstructed_convs.append(current_conv)
    
    print(f"Reconstructed {len(reconstructed_convs)} conversations")
    return reconstructed_convs

def format_mistral_finetune_data(conversations):
    """
    Format conversations for Mistral fine-tuning with the correct pattern:
    - No duplication of messages
    - Proper turn format
    """
    system_message = "You are a helpful and friendly chat assistant. Keep your responses natural and engaging."
    formatted_data = []
    
    for conv_idx, conv in enumerate(conversations):
        print(f"Processing conversation {conv_idx+1}/{len(conversations)}")
        
        # Process conversation as pairs of user/assistant messages
        for i in range(0, len(conv) - 1, 2):
            # If we don't have both user and assistant message, skip
            if i + 1 >= len(conv):
                continue
            
            user_message = conv[i]['question'].strip()
            assistant_message = conv[i]['answer'].strip()
            
            # Skip if messages are too short
            if len(user_message.split()) < 2 or len(assistant_message.split()) < 2:
                continue
            
            # Build conversation history up to this point
            history = []
            for j in range(0, i, 2):
                if j + 1 < len(conv):
                    prev_user = conv[j]['question'].strip()
                    prev_assistant = conv[j]['answer'].strip()
                    history.append((prev_user, prev_assistant))
            
            # Create the instruction string
            instruction = f"<s>[SYSTEM]{system_message}</s>\n"
            
            # Add conversation history
            for prev_user, prev_assistant in history:
                instruction += f"[INST] {prev_user} [/INST]{prev_assistant}</s>"
            
            # Add current user message
            instruction += f"[INST] {user_message} [/INST]"
            
            # Create the exchange
            exchange = {
                "instruction": instruction,
                "input": "",
                "output": f"{assistant_message}</s>"
            }
            
            formatted_data.append(exchange)
    
    return formatted_data

def validate_format(data):
    """
    Validate that there's no duplication in the format by checking if any assistant
    response appears as a user message in the next instruction.
    """
    issues_found = 0
    
    for i in range(len(data) - 1):
        current_output = data[i]["output"].replace("</s>", "").strip()
        next_instruction = data[i+1]["instruction"]
        
        if f"[INST] {current_output} [/INST]" in next_instruction:
            print(f"Issue #{issues_found+1}: Duplication found between items {i} and {i+1}")
            print(f"  Output: {current_output[:50]}...")
            issues_found += 1
            if issues_found >= 5:  # Limit number of issues to report
                print("Too many issues found, stopping validation.")
                break
    
    if issues_found == 0:
        print("Validation successful! No duplication found.")
        return True
    else:
        print(f"Validation failed with {issues_found} issues.")
        return False

def write_sample(data, count=3):
    """Write a sample of the data to a file for manual inspection"""
    sample_path = "data/sample_output.json"
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(data[:count], f, indent=2, ensure_ascii=False)
    print(f"Sample written to {sample_path}")

def main():
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading dataset...")
    dataset = load_dataset("SohamGhadge/casual-conversation")
    print(f"Dataset loaded. Examples: {len(dataset['train'])}")
    
    # Fix and reconstruct conversations from the dataset
    conversations = fix_dataset(dataset)
    
    # Format for Mistral fine-tuning
    formatted_data = format_mistral_finetune_data(conversations)
    print(f"Created {len(formatted_data)} formatted examples")
    
    # Validate format
    is_valid = validate_format(formatted_data)
    
    # Write sample for manual inspection
    write_sample(formatted_data)
    
    if is_valid:
        # Save formatted data
        output_path = os.path.join(output_dir, "chat_instructions.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to: {output_path}")
    else:
        print("\nNot saving due to validation issues. Please check the sample output.")

if __name__ == "__main__":
    main()
