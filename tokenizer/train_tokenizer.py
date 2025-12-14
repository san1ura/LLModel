"""
SentencePiece-based tokenizer module
Handles tokenization for the transformer model using SentencePiece
"""
import os
import torch
from typing import List, Union, Optional, Dict
from sentencepiece import SentencePieceProcessor


# For Python 3.8 compatibility
try:
    from typing import Literal
except ImportError:
    # For Python < 3.8, use typing_extensions
    from typing_extensions import Literal


class TokenizerTrainer:
    """
    Class for training SentencePiece tokenizers
    """
    def __init__(self, vocab_size: int = 32000, model_type: str = "BPE",
                 special_tokens: Optional[List[str]] = None):
        """
        Initialize SentencePiece tokenizer trainer
        
        Args:
            vocab_size: Size of vocabulary
            model_type: Type of model ('BPE', 'Unigram', 'WordPiece')
            special_tokens: List of special tokens
        """
        self.vocab_size = vocab_size
        # Convert model_type to uppercase for comparison
        model_type_upper = model_type.upper()
        self.model_type_map = {
            "BPE": "bpe",
            "UNIGRAM": "unigram",  # After .upper(), "Unigram" becomes "UNIGRAM"
            "WORDPIECE": "word",   # After .upper(), "WordPiece" becomes "WORDPIECE"
            # Also support already uppercase forms
            "WORD": "word"  # Common shorthand for WordPiece in SentencePiece
        }
        self.model_type = self.model_type_map.get(model_type_upper, "bpe")
        self.special_tokens = special_tokens or ["<pad>", "<unk>", "<s>", "</s>"]
        # Convert special tokens to format expected by SentencePiece
        self.sp_special_tokens = ",".join(self.special_tokens)

    def train_from_files(self, file_paths: List[str], output_path: str,
                         model_prefix: str = "spm_model",
                         character_coverage: float = 1.0,
                         input_sentence_size: int = -1,
                         shuffle_input_sentence: bool = True,
                         num_threads: int = 16,
                         **kwargs):
        """
        Train tokenizer from text files using SentencePiece

        Args:
            file_paths: List of file paths to use for training
            output_path: Path to save the trained tokenizer
            model_prefix: Prefix for the output model files
            character_coverage: Character coverage for SentencePiece
            input_sentence_size: Max number of sentences to use (-1 for all)
            shuffle_input_sentence: Whether to shuffle input sentences
            num_threads: Number of threads to use for training
            **kwargs: Additional arguments for SentencePiece training
        """
        # Combine all file paths into a single string
        input_files = ",".join(file_paths)
        
        # Prepare output directory
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Construct the model prefix (without extension)
        if output_path.endswith('.model'):
            model_prefix = output_path[:-6]  # Remove '.model' extension
        else:
            model_prefix = output_path
            
        # Training command parameters
        cmd = (
            f"--input={input_files} "
            f"--model_prefix={model_prefix} "
            f"--vocab_size={self.vocab_size} "
            f"--model_type={self.model_type} "
            f"--character_coverage={character_coverage} "
            f"--input_sentence_size={input_sentence_size} "
            f"--shuffle_input_sentence={shuffle_input_sentence} "
            f"--num_threads={num_threads}"
        )

        # SentencePiece already defines default special tokens (<unk>, <s>, </s>, <pad>)
        # We don't need to define them again which causes the conflict
        # Just make sure our trainer is configured properly
        # cmd += f" --user_defined_symbols={self.sp_special_tokens}"

        # Instead, we can set the special tokens using the parameters that SentencePiece supports
        cmd += f" --unk_piece=<unk> --bos_piece=<s> --eos_piece=</s> --pad_piece=<pad>"

        # Add any additional parameters
        for key, value in kwargs.items():
            cmd += f" --{key.replace('_', '-')}={value}"
        
        # Train the tokenizer
        import sentencepiece as spm
        spm.SentencePieceTrainer.Train(cmd)
        
        # Return tokenizer object
        return SentencePieceTokenizer.from_pretrained(output_path)

    def train_from_texts(self, texts: List[str], output_path: str,
                         model_prefix: str = "spm_model",
                         character_coverage: float = 1.0,
                         input_sentence_size: int = -1,
                         shuffle_input_sentence: bool = True,
                         num_threads: int = 16,
                         **kwargs):
        """
        Train tokenizer from list of texts using SentencePiece

        Args:
            texts: List of texts to use for training
            output_path: Path to save the trained tokenizer
            model_prefix: Prefix for the output model files
            character_coverage: Character coverage for SentencePiece
            input_sentence_size: Max number of sentences to use (-1 for all)
            shuffle_input_sentence: Whether to shuffle input sentences
            num_threads: Number of threads to use for training
            **kwargs: Additional arguments for SentencePiece training
        """
        import tempfile
        
        # Write texts to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            for text in texts:
                temp_file.write(text + '\n')
            temp_file_path = temp_file.name

        try:
            # Use the train_from_files method with the temp file
            result = self.train_from_files(
                [temp_file_path], 
                output_path, 
                model_prefix,
                character_coverage,
                input_sentence_size,
                shuffle_input_sentence,
                num_threads,
                **kwargs
            )
            return result
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)


class SentencePieceTokenizer:
    """
    Wrapper around SentencePiece for tokenization with PyTorch integration
    """
    def __init__(self, spm_model_path: str):
        """
        Initialize SentencePiece tokenizer

        Args:
            spm_model_path: Path to the SentencePiece model file
        """
        self.sp = SentencePieceProcessor()

        # SentencePiece creates .model file even if we provide a path with other extension
        # For example, if we pass "tokenizer.json", it creates "tokenizer.json.model"
        # So we check if the file exists as-is first, and if not, try appending .model
        if os.path.exists(spm_model_path):
            final_path = spm_model_path
        elif os.path.exists(spm_model_path + ".model"):
            final_path = spm_model_path + ".model"
        elif os.path.exists(os.path.splitext(spm_model_path)[0] + ".model"):
            final_path = os.path.splitext(spm_model_path)[0] + ".model"
        else:
            raise FileNotFoundError(f"SentencePiece model file not found at: {spm_model_path}")

        self.sp.load(final_path)
        self.spm_model_path = final_path

    @classmethod
    def from_pretrained(cls, path: str):
        """
        Load tokenizer from pretrained model file
        
        Args:
            path: Path to the SentencePiece model file
        """
        return cls(path)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add special tokens (BOS/EOS)

        Returns:
            List of token IDs
        """
        if add_special_tokens:
            return self.sp.encode(text, out_type=int)
        else:
            # Without special tokens
            return self.sp.encode(text, out_type=int, add_bos=False, add_eos=False)

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs to text

        Args:
            token_ids: List or tensor of token IDs to decode
            skip_special_tokens: Whether to skip special tokens during decoding

        Returns:
            Decoded text
        """
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()

        if skip_special_tokens:
            # Filter out special token IDs (common ones are 0, 1, 2)
            filtered_ids = [tid for tid in token_ids 
                           if tid != self.sp.pad_id() and 
                              tid != self.sp.bos_id() and 
                              tid != self.sp.eos_id()]
            return self.sp.decode(filtered_ids)
        else:
            return self.sp.decode(token_ids)

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of texts

        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token ID lists
        """
        if add_special_tokens:
            return [self.sp.encode(text, out_type=int) for text in texts]
        else:
            return [self.sp.encode(text, out_type=int, add_bos=False, add_eos=False) for text in texts]

    def batch_decode(self, batch_token_ids: Union[List[List[int]], torch.Tensor], 
                     skip_special_tokens: bool = False) -> List[str]:
        """
        Decode a batch of token ID lists

        Args:
            batch_token_ids: Batch of token ID lists or tensor
            skip_special_tokens: Whether to skip special tokens during decoding

        Returns:
            List of decoded texts
        """
        if torch.is_tensor(batch_token_ids):
            batch_token_ids = batch_token_ids.tolist()

        if skip_special_tokens:
            decoded_texts = []
            for token_ids in batch_token_ids:
                # Filter out special token IDs (common ones are 0, 1, 2)
                filtered_ids = [tid for tid in token_ids 
                               if tid != self.sp.pad_id() and 
                                  tid != self.sp.bos_id() and 
                                  tid != self.sp.eos_id()]
                decoded_texts.append(self.sp.decode(filtered_ids))
            return decoded_texts
        else:
            return [self.sp.decode(token_ids) for token_ids in batch_token_ids]

    def get_vocab_size(self) -> int:
        """Get the vocabulary size"""
        return self.sp.get_piece_size()

    def get_vocab(self) -> dict:
        """Get the vocabulary as a dictionary {token: id}"""
        vocab = {}
        for i in range(self.get_vocab_size()):
            token = self.sp.id_to_piece(i)
            vocab[token] = i
        return vocab

    def pad_token_id(self):
        """Return the padding token ID"""
        return self.sp.pad_id()

    def bos_token_id(self):
        """Return the beginning-of-sentence token ID"""
        return self.sp.bos_id()

    def eos_token_id(self):
        """Return the end-of-sentence token ID"""
        return self.sp.eos_id()

    def unk_token_id(self):
        """Return the unknown token ID"""
        return self.sp.unk_id()

    def save(self, path: str):
        """
        Save the tokenizer model file to the specified path.
        Note: Since SentencePiece models are binary, the model file itself contains all information.
        """
        # Copy the model file to the new location
        import shutil
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # For SentencePiece, the path to the .model file should be correct
        # Check if source file exists before copying
        if os.path.exists(self.spm_model_path):
            shutil.copy2(self.spm_model_path, path)
        else:
            # If the direct model file doesn't exist, try model_path without extension
            # SentencePiece might have saved different files
            model_path_no_ext = os.path.splitext(self.spm_model_path)[0]
            model_files = [f"{model_path_no_ext}.model", f"{model_path_no_ext}.vocab"]
            for model_file in model_files:
                if os.path.exists(model_file):
                    shutil.copy2(model_file, path)
                    return
            # If none of the expected files exist, raise error
            raise FileNotFoundError(f"Model file does not exist: {self.spm_model_path}")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens to a single string

        Args:
            tokens: List of tokens to join

        Returns:
            Joined string
        """
        # SentencePiece handles detokenization internally, but if we have tokens as separate elements,
        # we'll try to join them with spaces (this is a simplification)
        return " ".join(tokens)


def create_tokenizer_from_vocab(vocab: Union[Dict[str, int], List[str]],
                               output_path: str,
                               special_tokens: Optional[List[str]] = None) -> SentencePieceTokenizer:
    """
    Create a SentencePiece tokenizer from a predefined vocabulary
    Note: SentencePiece is designed to learn vocabulary from raw text, not from a predefined vocab.
    This function creates a temporary text file to train on the vocabulary tokens.

    Args:
        vocab: Vocabulary as dictionary {token: id} or list of tokens
        output_path: Path to save the tokenizer
        special_tokens: List of special tokens to include

    Returns:
        SentencePieceTokenizer instance
    """
    import tempfile

    # Convert list to dict if necessary
    if isinstance(vocab, list):
        vocab_items = vocab
    else:
        # Sort by ID to maintain order
        vocab_items = [item[0] for item in sorted(vocab.items(), key=lambda x: x[1])]

    # Create a larger temporary file with vocabulary items repeated multiple times
    # to ensure SentencePiece can create a proper model
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        # Repeat each token in the vocab multiple times to create more data
        for _ in range(100):  # Repeat 100 times to have enough data
            for token in vocab_items:
                # Add special handling for special tokens
                temp_file.write(f"{token}\n")
        temp_file_path = temp_file.name

    try:
        # Create a trainer with sufficient vocab size
        # Ensure the vocab size is appropriate for the training data
        base_vocab_size = len(vocab_items) + (len(special_tokens) if special_tokens else 4)
        # Use a vocab size slightly larger than the base to account for subword units
        # But not too large to cause issues with small training data
        vocab_size_needed = base_vocab_size + 10  # Add small buffer

        trainer = TokenizerTrainer(
            vocab_size=vocab_size_needed,
            special_tokens=special_tokens
        )

        tokenizer = trainer.train_from_files(
            [temp_file_path],
            output_path
        )
        return tokenizer
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


def train_default_tokenizer(data_paths: List[str], output_path: str,
                          vocab_size: int = 32000) -> SentencePieceTokenizer:
    """
    Train a default SentencePiece tokenizer on the given data

    Args:
        data_paths: List of paths to training data files
        output_path: Path to save the trained tokenizer
        vocab_size: Size of the vocabulary to create

    Returns:
        SentencePieceTokenizer instance
    """
    trainer = TokenizerTrainer(vocab_size=vocab_size)
    return trainer.train_from_files(data_paths, output_path)