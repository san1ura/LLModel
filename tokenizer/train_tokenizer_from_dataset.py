"""
Train SentencePiece tokenizer from the LMSYS dataset
"""

import os
import sys
import tempfile

from datasets import load_dataset
from tqdm import tqdm

# Add the project root to Python path to allow direct imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Try to import the SentencePiece trainer
try:
    from tokenizer.custom_sentencepiece_tokenizer import \
        train_sentencepiece_tokenizer

    USE_SENTENCEPIECE = True
    print("Using SentencePiece tokenizer for training")
except ImportError:
    from tokenizer.train_tokenizer import TokenizerTrainer

    USE_SENTENCEPIECE = False
    print("Using tokenizers library for training")


# custom dataset for test!
def train_tokenizer_from_lmsys_dataset(
    dataset_name="ytz20/LMSYS-Chat-GPT-5-Chat-Response",
    output_path="weights/tokenizer.model",
    vocab_size=32000,
):
    """
    Train a tokenizer from the LMSYS dataset

    Args:
        dataset_name: Name of the dataset to use for training
        output_path: Path to save the trained tokenizer (without extension for SentencePiece)
        vocab_size: Size of the vocabulary
    """
    print(f"Loading dataset: {dataset_name}")

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Collect all text data from the dataset
    print("Collecting text data from dataset...")
    all_texts = []

    # Iterate over all splits in the dataset
    for split_name in dataset.keys():
        split = dataset[split_name]

        # Process each sample in the split
        for i in tqdm(range(len(split)), desc=f"Processing {split_name}"):
            row = split[i]

            # Extract text from different possible structures in the dataset
            if isinstance(row, dict):
                for key, value in row.items():
                    if isinstance(value, str):
                        all_texts.append(value)
                    elif isinstance(value, list):
                        # Handle nested structures like conversations
                        for item in value:
                            if isinstance(item, dict):
                                for sub_key, sub_value in item.items():
                                    if isinstance(sub_value, str):
                                        all_texts.append(sub_value)
                            elif isinstance(item, str):
                                all_texts.append(item)
            elif isinstance(row, str):
                all_texts.append(row)

            # Limit the number of samples during the initial processing to avoid memory issues
            if (
                len(all_texts) >= 50000
            ):  # Increase from 10k to 50k for better vocabulary
                break

        if len(all_texts) >= 50000:
            break

    print(f"Collected {len(all_texts)} text samples for tokenizer training")

    # Create temporary file to store all texts
    temp_path = output_path + "_temp_training.txt"
    with open(temp_path, "w", encoding="utf-8") as temp_file:
        for text in all_texts:
            # Write each text as a separate line
            temp_file.write(
                text.replace("\n", " ") + "\n"
            )  # Replace newlines with spaces to keep each text on one line

    try:
        # Train tokenizer
        if USE_SENTENCEPIECE:
            print("Training SentencePiece tokenizer...")
            # For SentencePiece, we need to provide the path without extension
            output_path_no_ext = (
                output_path.rsplit(".", 1)[0] if "." in output_path else output_path
            )
            trained_tokenizer = train_sentencepiece_tokenizer(
                data_paths=[temp_path],
                model_path=output_path_no_ext,
                vocab_size=vocab_size,
                model_type="bpe",  # Use BPE model type
            )
            print(
                f"SentencePiece tokenizer trained successfully and saved to {output_path_no_ext}.model"
            )
        else:
            print("Training tokenizers library tokenizer...")
            trainer = TokenizerTrainer(
                vocab_size=vocab_size,
                model_type="BPE",
                special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
            )

            trained_tokenizer = trainer.train_from_files(
                file_paths=[temp_path],
                output_path=output_path,
                min_frequency=2,  # Reduced min frequency for better coverage
                show_progress=True,
            )
            print(f"Tokenizer trained successfully and saved to {output_path}")

        return trained_tokenizer

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    train_tokenizer_from_lmsys_dataset()
