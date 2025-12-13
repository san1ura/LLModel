"""
Tokenizer training and management module
Handles tokenization for the transformer model
"""
import os
import json
from typing import List, Union, Optional, Dict
import torch

# Prefer SentencePiece if available, but always make tokenizers library available
try:
    from tokenizer.custom_sentencepiece_tokenizer import SentencePieceTokenizer, train_sentencepiece_tokenizer
    HAS_SENTENCEPIECE = True
    # Even if SentencePiece is available, we still need tokenizers for fallback
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
    from tokenizers.processors import TemplateProcessing
except ImportError:
    HAS_SENTENCEPIECE = False
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
    from tokenizers.processors import TemplateProcessing


class TokenizerTrainer:
    """
    Class for training tokenizers with different algorithms
    """
    def __init__(self, vocab_size: int = 32000, model_type: str = "BPE",
                 special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.special_tokens = special_tokens or ["<pad>", "<unk>", "<s>", "</s>"]

        # Only create the traditional tokenizer if SentencePiece is NOT available
        if not HAS_SENTENCEPIECE:
            # Initialize tokenizer based on model type
            if model_type == "BPE":
                self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
            elif model_type == "Unigram":
                self.tokenizer = Tokenizer(models.Unigram())
            elif model_type == "WordPiece":
                self.tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Set up normalizers and pre-tokenizers
            self.tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFKC(),
                normalizers.Lowercase()
            ])

            # Use ByteLevel pre-tokenizer which is more common in modern tokenizers
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

            # Set up post-processor to handle special tokens
            self.tokenizer.post_processor = TemplateProcessing(
                single="<s> $A </s>",
                special_tokens=[
                    ("<s>", 1),
                    ("</s>", 2),
                ]
            )
        else:
            # If SentencePiece is available, this trainer should not be used
            # However, to maintain compatibility, we'll allow it but issue a warning
            import warnings
            warnings.warn(
                "SentencePiece is available but TokenizerTrainer is being used. "
                "For best results, use train_sentencepiece_tokenizer() instead.",
                UserWarning
            )
            # Initialize tokenizer based on model type (fallback to tokenizers library)
            if model_type == "BPE":
                self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
            elif model_type == "Unigram":
                self.tokenizer = Tokenizer(models.Unigram())
            elif model_type == "WordPiece":
                self.tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Set up normalizers and pre-tokenizers
            self.tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFKC(),
                normalizers.Lowercase()
            ])

            # Use ByteLevel pre-tokenizer which is more common in modern tokenizers
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

            # Set up post-processor to handle special tokens
            self.tokenizer.post_processor = TemplateProcessing(
                single="<s> $A </s>",
                special_tokens=[
                    ("<s>", 1),
                    ("</s>", 2),
                ]
            )
    
    def train_from_files(self, file_paths: List[str], output_path: str,
                         min_frequency: int = 2, show_progress: bool = True):
        """
        Train tokenizer from text files

        Args:
            file_paths: List of file paths to use for training
            output_path: Path to save the trained tokenizer
            min_frequency: Minimum frequency for a token to be included
            show_progress: Whether to show training progress
        """
        if HAS_SENTENCEPIECE:
            import warnings
            warnings.warn(
                "SentencePiece is available but TokenizerTrainer.train_from_files is being used. "
                "For best results, use train_sentencepiece_tokenizer() instead.",
                UserWarning
            )

        # Create trainer based on model type
        if self.model_type == "BPE":
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=min_frequency,
                special_tokens=self.special_tokens,
                show_progress=show_progress
            )
        elif self.model_type == "Unigram":
            trainer = trainers.UnigramTrainer(
                vocab_size=self.vocab_size,
                special_tokens=self.special_tokens,
                show_progress=show_progress
            )
        elif self.model_type == "WordPiece":
            trainer = trainers.WordPieceTrainer(
                vocab_size=self.vocab_size,
                min_frequency=min_frequency,
                special_tokens=self.special_tokens,
                show_progress=show_progress
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Train the tokenizer
        self.tokenizer.train(file_paths, trainer)

        # Save tokenizer
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.tokenizer.save(output_path)

        return self.tokenizer
    
    def train_from_texts(self, texts: List[str], output_path: str,
                         min_frequency: int = 2, show_progress: bool = True):
        """
        Train tokenizer from list of texts

        Args:
            texts: List of texts to use for training
            output_path: Path to save the trained tokenizer
            min_frequency: Minimum frequency for a token to be included
            show_progress: Whether to show training progress
        """
        if HAS_SENTENCEPIECE:
            import warnings
            warnings.warn(
                "SentencePiece is available but TokenizerTrainer.train_from_texts is being used. "
                "For best results, use train_sentencepiece_tokenizer() instead.",
                UserWarning
            )

        # Write texts to temporary files
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file = os.path.join(tmp_dir, "temp_training.txt")
            with open(temp_file, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(text + "\n")

            return self.train_from_files([temp_file], output_path, min_frequency, show_progress)


class TokenizerWrapper:
    """
    Wrapper around either SentencePiece or tokenizers library with PyTorch integration
    """
    def __init__(self, tokenizer_path: str = None, tokenizer_obj=None):
        if HAS_SENTENCEPIECE:
            if tokenizer_path:
                # Determine the file type by extension or content
                # SentencePiece models typically have .model extension
                if tokenizer_path.endswith('.model'):
                    # Try to load as SentencePiece model
                    try:
                        self.tokenizer = SentencePieceTokenizer.from_pretrained(tokenizer_path)
                        # Use the SentencePiece tokenizer's token IDs
                        self.pad_token_id = self.tokenizer.pad_token_id
                        self.bos_token_id = self.tokenizer.bos_token_id
                        self.eos_token_id = self.tokenizer.eos_token_id
                        self.unk_token_id = self.tokenizer.unk_token_id
                    except Exception:
                        # If SentencePiece fails, fall back to tokenizers library
                        self.tokenizer = Tokenizer.from_file(tokenizer_path)

                        # Set up token IDs by looking up in the vocabulary
                        vocab = self.tokenizer.get_vocab()

                        # Define special token IDs by looking them up in the vocabulary
                        self.pad_token_id = vocab.get("<pad>", 0)  # Default to 0 if not found
                        self.bos_token_id = vocab.get("<s>", 1)    # Default to 1 if not found
                        self.eos_token_id = vocab.get("</s>", 2)   # Default to 2 if not found
                        self.unk_token_id = vocab.get("<unk>", 3)  # Default to 3 if not found
                else:
                    # For .json files or other extensions, use tokenizers library
                    self.tokenizer = Tokenizer.from_file(tokenizer_path)

                    # Set up token IDs by looking up in the vocabulary
                    vocab = self.tokenizer.get_vocab()

                    # Define special token IDs by looking them up in the vocabulary
                    self.pad_token_id = vocab.get("<pad>", 0)  # Default to 0 if not found
                    self.bos_token_id = vocab.get("<s>", 1)    # Default to 1 if not found
                    self.eos_token_id = vocab.get("</s>", 2)   # Default to 2 if not found
                    self.unk_token_id = vocab.get("<unk>", 3)  # Default to 3 if not found
            elif tokenizer_obj:
                self.tokenizer = tokenizer_obj
                # Check if tokenizer_obj is actually a SentencePieceTokenizer or a tokenizers object
                # The isinstance check should work properly
                if isinstance(self.tokenizer, SentencePieceTokenizer):
                    self.pad_token_id = self.tokenizer.pad_token_id
                    self.bos_token_id = self.tokenizer.bos_token_id
                    self.eos_token_id = self.tokenizer.eos_token_id
                    self.unk_token_id = self.tokenizer.unk_token_id
                else:
                    # This is a tokenizers library object (e.g., created by TokenizerTrainer)
                    vocab = self.tokenizer.get_vocab()
                    self.pad_token_id = vocab.get("<pad>", 0)
                    self.bos_token_id = vocab.get("<s>", 1)
                    self.eos_token_id = vocab.get("</s>", 2)
                    self.unk_token_id = vocab.get("<unk>", 3)
            else:
                raise ValueError("Either tokenizer_path or tokenizer_obj must be provided")
        else:
            if tokenizer_path:
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
            elif tokenizer_obj:
                self.tokenizer = tokenizer_obj
            else:
                raise ValueError("Either tokenizer_path or tokenizer_obj must be provided")

            # Set up token IDs by looking up in the vocabulary
            vocab = self.tokenizer.get_vocab()

            # Define special token IDs by looking them up in the vocabulary
            self.pad_token_id = vocab.get("<pad>", 0)  # Default to 0 if not found
            self.bos_token_id = vocab.get("<s>", 1)    # Default to 1 if not found
            self.eos_token_id = vocab.get("</s>", 2)   # Default to 2 if not found
            self.unk_token_id = vocab.get("<unk>", 3)  # Default to 3 if not found
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add special tokens (BOS/EOS)

        Returns:
            List of token IDs
        """
        if HAS_SENTENCEPIECE and isinstance(self.tokenizer, SentencePieceTokenizer):
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        else:
            if add_special_tokens:
                encoding = self.tokenizer.encode(text)
            else:
                # Temporarily disable post-processing to avoid adding special tokens
                original_post_processor = self.tokenizer.post_processor
                self.tokenizer.post_processor = None
                encoding = self.tokenizer.encode(text)
                self.tokenizer.post_processor = original_post_processor

            # The tokenizers library's encode method returns an Encoding object
            # We need to extract the IDs from it
            if hasattr(encoding, 'ids'):
                return encoding.ids
            else:
                # If it's already a list (e.g., from SentencePiece wrapper)
                return encoding

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

        if HAS_SENTENCEPIECE:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of texts

        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token ID lists
        """
        if HAS_SENTENCEPIECE and isinstance(self.tokenizer, SentencePieceTokenizer):
            return self.tokenizer.batch_encode(texts, add_special_tokens=add_special_tokens)
        else:
            encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
            # The tokenizers library's encode_batch returns a list of Encoding objects
            # We need to extract the IDs from each
            result = []
            for enc in encodings:
                if hasattr(enc, 'ids'):
                    result.append(enc.ids)
                else:
                    result.append(enc)
            return result

    def batch_decode(self, batch_token_ids: Union[List[List[int]], torch.Tensor]) -> List[str]:
        """
        Decode a batch of token ID lists

        Args:
            batch_token_ids: Batch of token ID lists or tensor

        Returns:
            List of decoded texts
        """
        if torch.is_tensor(batch_token_ids):
            batch_token_ids = batch_token_ids.tolist()

        if HAS_SENTENCEPIECE:
            return self.tokenizer.batch_decode(batch_token_ids)
        else:
            return self.tokenizer.decode_batch(batch_token_ids)

    def get_vocab_size(self) -> int:
        """Get the vocabulary size"""
        if HAS_SENTENCEPIECE:
            return self.tokenizer.get_vocab_size()
        else:
            return self.tokenizer.get_vocab_size()

    def get_vocab(self) -> dict:
        """Get the vocabulary as a dictionary {token: id}"""
        if HAS_SENTENCEPIECE:
            return self.tokenizer.get_vocab()
        else:
            return self.tokenizer.get_vocab()

    def save(self, path: str):
        """Save the tokenizer to a file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Check the actual type of tokenizer object, not just HAS_SENTENCEPIECE
        if isinstance(self.tokenizer, SentencePieceTokenizer):
            # For SentencePiece tokenizer, it doesn't directly support JSON format like tokenizers library
            # In practice, we usually just note that it should be saved as .model
            pass  # SentencePiece models should be saved during training, not here
        else:
            # This is a tokenizers library tokenizer, so use its save method
            # This handles both the case when HAS_SENTENCEPIECE=False and
            # when HAS_SENTENCEPIECE=True but tokenizer is still from tokenizers library
            # (e.g., created by TokenizerTrainer)
            self.tokenizer.save(path)

    @classmethod
    def from_pretrained(cls, path: str):
        """Load tokenizer from a pre-trained file"""
        return cls(tokenizer_path=path)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens to a single string

        Args:
            tokens: List of tokens to join

        Returns:
            Joined string
        """
        if HAS_SENTENCEPIECE:
            return self.tokenizer.convert_tokens_to_string(tokens)
        else:
            # We'll join tokens with a space, which is a simple approach
            # For more sophisticated joining, we might need to consider
            # the specific tokenization scheme
            return " ".join(tokens)


def create_tokenizer_from_vocab(vocab: Union[Dict[str, int], List[str]], 
                               output_path: str,
                               special_tokens: Optional[List[str]] = None) -> TokenizerWrapper:
    """
    Create a tokenizer from a predefined vocabulary
    
    Args:
        vocab: Vocabulary as dictionary {token: id} or list of tokens
        output_path: Path to save the tokenizer
        special_tokens: List of special tokens to include
        
    Returns:
        TokenizerWrapper instance
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    import tempfile
    import json
    
    # Convert list to dict if necessary
    if isinstance(vocab, list):
        vocab = {token: idx for idx, token in enumerate(vocab)}
    
    # Create vocabulary files needed by tokenizers library
    with tempfile.TemporaryDirectory() as tmp_dir:
        vocab_file = os.path.join(tmp_dir, "vocab.json")
        merges_file = os.path.join(tmp_dir, "merges.txt")
        
        # Save vocabulary
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2)
        
        # Create empty merges file for BPE (since we're creating from vocab)
        with open(merges_file, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")
        
        # Create tokenizer from files using from_file method
        bpe_model = BPE.from_file(vocab_file, merges_file)
        tokenizer = Tokenizer(bpe_model)
        bpe_model.unk_token = "<unk>"
        
        # Add the special tokens
        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)
        
        # Save tokenizer
        tokenizer.save(output_path)
        
        return TokenizerWrapper(tokenizer_path=output_path)


def train_default_tokenizer(data_paths: List[str], output_path: str, 
                          vocab_size: int = 32000) -> TokenizerWrapper:
    """
    Train a default tokenizer on the given data
    
    Args:
        data_paths: List of paths to training data files
        output_path: Path to save the trained tokenizer
        vocab_size: Size of the vocabulary to create
        
    Returns:
        TokenizerWrapper instance
    """
    trainer = TokenizerTrainer(vocab_size=vocab_size)
    trainer.train_from_files(data_paths, output_path)
    return TokenizerWrapper(tokenizer_path=output_path)