"""
Prepare LMSYS dataset for model training
This script converts the dataset to the format expected by the model
"""
import os
import torch
import json
import sys
import os

# Add the project root to Python path to allow direct imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from datasets import load_dataset
from tqdm import tqdm
from tokenizer.train_tokenizer import TokenizerWrapper
import random


def prepare_lmsys_dataset_for_training(
    dataset_name="ytz20/LMSYS-Chat-GPT-5-Chat-Response",
    tokenizer_path="weights/tokenizer.model",
    output_path="data/preprocessed/lmsys_dataset.bin",
    max_length=2048
):
    """
    Prepare the LMSYS dataset for model training

    Args:
        dataset_name: Name of the dataset to use for training
        tokenizer_path: Path to the trained tokenizer
        output_path: Path to save the processed dataset
        max_length: Maximum sequence length
    """
    print(f"Loading dataset: {dataset_name}")

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Load the tokenizer
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = TokenizerWrapper.from_pretrained(tokenizer_path)

    # Prepare output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect all text data from the dataset
    print("Processing dataset for training...")

    # Open output file for writing
    with open(output_path, 'wb') as out_f:
        processed_count = 0

        # Iterate over all splits in the dataset
        for split_name in dataset.keys():
            split = dataset[split_name]

            # Process each sample in the split
            for i in tqdm(range(len(split)), desc=f"Processing {split_name}"):
                row = split[i]

                # Convert the row to text format
                # The LMSYS dataset may have different structures, so try different approaches
                text = ""

                # Try to find conversation or content fields
                if isinstance(row, dict):
                    if 'conversations' in row:
                        # Handle conversation format - typical in chat datasets
                        conversations = row['conversations']
                        if isinstance(conversations, list):
                            for conv in conversations:
                                if isinstance(conv, dict):
                                    if 'content' in conv:
                                        text += conv['content'] + "\n"
                                    elif 'value' in conv:
                                        text += conv['value'] + "\n"
                                elif isinstance(conv, str):
                                    text += conv + "\n"
                    elif 'content' in row:
                        # Handle content field
                        text = str(row['content'])
                    elif 'text' in row:
                        # Handle text field
                        text = str(row['text'])
                    elif 'prompt' in row:
                        # Handle prompt/response format
                        prompt = str(row['prompt']) if 'prompt' in row else ""
                        response = str(row['response']) if 'response' in row else ""
                        text = f"Prompt: {prompt}\nResponse: {response}\n"
                    else:
                        # Try to concatenate all string fields
                        for key, value in row.items():
                            if isinstance(value, str):
                                text += str(value) + "\n"
                elif isinstance(row, str):
                    # If the row itself is a string
                    text = row
                else:
                    # Convert other types to string
                    text = str(row)

                if text.strip():
                    # Tokenize the text
                    tokens = tokenizer.encode(text, add_special_tokens=False)

                    # Add special tokens if needed (we'll add BOS and EOS tokens)
                    bos_token_id = tokenizer.tokenizer.token_to_id('<s>')
                    eos_token_id = tokenizer.tokenizer.token_to_id('</s>')

                    if bos_token_id is not None and eos_token_id is not None:
                        tokens = [bos_token_id] + tokens + [eos_token_id]
                    elif bos_token_id is not None:
                        tokens = [bos_token_id] + tokens
                    elif eos_token_id is not None:
                        tokens = tokens + [eos_token_id]

                    # Split into chunks of max_length if needed
                    for i in range(0, len(tokens), max_length):
                        chunk = tokens[i:i + max_length]

                        # Pad if necessary
                        pad_token_id = tokenizer.tokenizer.token_to_id('<pad>')
                        if pad_token_id is None:
                            pad_token_id = 0  # Default to 0 if <pad> token not found

                        if len(chunk) < max_length:
                            chunk.extend([pad_token_id] * (max_length - len(chunk)))

                        # Write to binary file with length prefix
                        seq_len = len(chunk)
                        out_f.write(seq_len.to_bytes(4, 'little'))
                        out_f.write(torch.tensor(chunk, dtype=torch.long).numpy().tobytes())

                        processed_count += 1

                        # For large datasets, we might want to limit the number of samples during initial testing
                        # Commenting this out for now to process the full dataset
                        # if processed_count >= 50000:  # Limit to 50k samples initially
                        #     print(f"Processed {processed_count} samples (limit reached)")
                        #     return output_path

    print(f"Dataset processed successfully. Total samples: {processed_count}")
    return output_path


def prepare_mixed_datasets(
    dataset_configs,
    tokenizer_path="weights/tokenizer.model",
    output_path="data/preprocessed/mixed_dataset.bin",
    max_length=2048,
    mixing_ratios=None
):
    """
    Prepare a mixed dataset by combining multiple datasets according to specified ratios

    Args:
        dataset_configs: List of tuples (dataset_name, process_function),
                        where process_function is a function that extracts text from the dataset
        tokenizer_path: Path to the trained tokenizer
        output_path: Path to save the processed mixed dataset
        max_length: Maximum sequence length
        mixing_ratios: List of ratios for each dataset (should sum to 1.0).
                       If None, datasets will be mixed equally
    """
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = TokenizerWrapper.from_pretrained(tokenizer_path)

    # Prepare output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare list of all tokenized sequences from all datasets
    all_sequences = []

    # Calculate mixing ratios if not provided
    if mixing_ratios is None:
        mixing_ratios = [1.0 / len(dataset_configs)] * len(dataset_configs)

    # Load and process each dataset
    for i, (dataset_name, process_func) in enumerate(dataset_configs):
        print(f"Processing dataset {i+1}/{len(dataset_configs)}: {dataset_name}")

        # Calculate how many samples to take from this dataset based on the ratio
        expected_samples = int(mixing_ratios[i] * 10000)  # Use 10000 as baseline for calculation

        # Load the dataset
        dataset = load_dataset(dataset_name)

        dataset_sequences = []
        processed_count = 0

        # Process each sample in the dataset
        for split_name in dataset.keys():
            if processed_count >= expected_samples:
                break

            split = dataset[split_name]

            for j in tqdm(range(len(split)), desc=f"Processing {split_name} from {dataset_name}"):
                if processed_count >= expected_samples:
                    break

                row = split[j]

                # Use the process function to extract text
                text = process_func(row)

                if text and text.strip():
                    # Tokenize the text
                    tokens = tokenizer.encode(text, add_special_tokens=False)

                    # Add special tokens if needed (we'll add BOS and EOS tokens)
                    bos_token_id = tokenizer.tokenizer.token_to_id('<s>')
                    eos_token_id = tokenizer.tokenizer.token_to_id('</s>')

                    if bos_token_id is not None and eos_token_id is not None:
                        tokens = [bos_token_id] + tokens + [eos_token_id]
                    elif bos_token_id is not None:
                        tokens = [bos_token_id] + tokens
                    elif eos_token_id is not None:
                        tokens = tokens + [eos_token_id]

                    # Split into chunks of max_length if needed
                    for k in range(0, len(tokens), max_length):
                        chunk = tokens[k:k + max_length]

                        # Pad if necessary
                        pad_token_id = tokenizer.tokenizer.token_to_id('<pad>')
                        if pad_token_id is None:
                            pad_token_id = 0  # Default to 0 if <pad> token not found

                        if len(chunk) < max_length:
                            chunk.extend([pad_token_id] * (max_length - len(chunk)))

                        dataset_sequences.append(chunk)
                        processed_count += 1

                        if processed_count >= expected_samples:
                            break

        all_sequences.extend(dataset_sequences)
        print(f"Dataset {dataset_name} contributed {len(dataset_sequences)} sequences")

    # Shuffle the combined sequences for proper mixing
    print("Shuffling combined dataset...")
    random.shuffle(all_sequences)

    # Write the mixed dataset to file
    print(f"Writing mixed dataset to: {output_path}")
    with open(output_path, 'wb') as out_f:
        for i, tokens in enumerate(tqdm(all_sequences, desc="Writing sequences")):
            # Write to binary file with length prefix
            seq_len = len(tokens)
            out_f.write(seq_len.to_bytes(4, 'little'))
            out_f.write(torch.tensor(tokens, dtype=torch.long).numpy().tobytes())

    print(f"Mixed dataset prepared successfully. Total samples: {len(all_sequences)}")
    return output_path


def load_and_test_dataset(dataset_path, tokenizer_path, num_samples=5):
    """
    Test the prepared dataset by loading some samples
    """
    from data.preprocessed.build_dataset import PreprocessedDataset
    
    print(f"Testing dataset from: {dataset_path}")
    
    # Load tokenizer
    tokenizer = TokenizerWrapper.from_pretrained(tokenizer_path)
    
    # Load dataset
    dataset = PreprocessedDataset(dataset_path, block_size=1024)  # Using smaller block for testing
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Test a few samples
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_ids = sample['input_ids']
        labels = sample['labels']
        
        # Decode the text
        decoded_text = tokenizer.decode(input_ids[input_ids != 0])  # Assuming 0 is pad token
        
        print(f"\nSample {i+1}:")
        print(f"Input shape: {input_ids.shape}")
        print(f"First 100 chars of decoded text: {decoded_text[:100]}...")
        
    return dataset


def process_lmsys_conversation(row):
    """Extract text from LMSYS conversation format"""
    text = ""

    if isinstance(row, dict):
        if 'conversations' in row:
            # Handle conversation format - typical in chat datasets
            conversations = row['conversations']
            if isinstance(conversations, list):
                for conv in conversations:
                    if isinstance(conv, dict):
                        if 'content' in conv:
                            text += conv['content'] + "\n"
                        elif 'value' in conv:
                            text += conv['value'] + "\n"
                    elif isinstance(conv, str):
                        text += conv + "\n"
        elif 'content' in row:
            # Handle content field
            text = str(row['content'])
        elif 'text' in row:
            # Handle text field
            text = str(row['text'])
        elif 'prompt' in row:
            # Handle prompt/response format
            prompt = str(row['prompt']) if 'prompt' in row else ""
            response = str(row['response']) if 'response' in row else ""
            text = f"Prompt: {prompt}\nResponse: {response}\n"
        else:
            # Try to concatenate all string fields
            for key, value in row.items():
                if isinstance(value, str):
                    text += str(value) + "\n"
    elif isinstance(row, str):
        # If the row itself is a string
        text = row
    else:
        # Convert other types to string
        text = str(row)

    return text.strip()


def process_text_document(row):
    """Extract text from a text document format"""
    text = ""

    if isinstance(row, dict):
        if 'text' in row:
            text = str(row['text'])
        elif 'content' in row:
            text = str(row['content'])
        elif 'document' in row:
            text = str(row['document'])
        else:
            # Concatenate all string values
            for key, value in row.items():
                if isinstance(value, str):
                    text += str(value) + "\n"
    elif isinstance(row, str):
        text = row
    else:
        text = str(row)

    return text.strip()


def prepare_mixed_dataset_example():
    """Example function to demonstrate mixed dataset preparation"""
    # Define dataset configurations
    dataset_configs = [
        ("ytz20/LMSYS-Chat-GPT-5-Chat-Response", process_lmsys_conversation),  # Chat dialogues
        # Add more datasets as needed - example below
        # ("wikitext", process_text_document),  # General text (uncomment to use)
        # ("codeparrot/small-v2", process_text_document),  # Code snippets
        # ("eli5", process_text_document),  # Explain like I'm 5
    ]

    # Define mixing ratios (should sum to 1.0)
    # Example: 60% LMSYS chat, 40% general text
    # mixing_ratios = [0.6, 0.4]  # Uncomment and adjust when adding more datasets

    # For now, using just one dataset, will use default equal ratios
    mixing_ratios = None

    # Prepare the mixed dataset
    output_path = prepare_mixed_datasets(
        dataset_configs=dataset_configs,
        tokenizer_path="weights/tokenizer.model",
        output_path="data/preprocessed/mixed_dataset.bin",
        max_length=2048,
        mixing_ratios=mixing_ratios
    )

    # Test the mixed dataset
    test_dataset = load_and_test_dataset(output_path, "weights/tokenizer.model")

    return output_path


if __name__ == "__main__":
    # First, make sure tokenizer exists
    tokenizer_path = "weights/tokenizer.model"
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        print("Please run train_tokenizer_from_dataset.py first or provide a tokenizer path")
        exit(1)
    
    # Prepare the dataset
    output_path = prepare_lmsys_dataset_for_training()
    
    # Test the dataset
    test_dataset = load_and_test_dataset(output_path, tokenizer_path)