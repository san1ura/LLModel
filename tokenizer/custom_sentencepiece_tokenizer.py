"""
Custom SentencePiece tokenizer implementation for the transformer model
"""
import os
import sentencepiece as spm
from typing import List, Union, Optional
import torch
import json


class SentencePieceTokenizer:
    """
    Custom SentencePiece tokenizer wrapper for the transformer model
    """
    def __init__(self, model_path: str = None, sp_model: spm.SentencePieceProcessor = None):
        if model_path:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
        elif sp_model:
            self.sp = sp_model
        else:
            raise ValueError("Either model_path or sp_model must be provided")
        
        # Set up token IDs
        self.unk_token_id = self.sp.unk_id()
        self.pad_token_id = self.sp.pad_id() if self.sp.pad_id() != -1 else 0
        self.bos_token_id = self.sp.bos_id() if self.sp.bos_id() != -1 else 1
        self.eos_token_id = self.sp.eos_id() if self.sp.eos_id() != -1 else 2
    
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
            return self.sp.encode(text, out_type=int, add_bos=True, add_eos=True)
        else:
            return self.sp.encode(text, out_type=int)
    
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
        
        return self.sp.decode(token_ids, remove_extra_whitespaces=skip_special_tokens)
    
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
            return [self.sp.encode(text, out_type=int, add_bos=True, add_eos=True) for text in texts]
        else:
            return [self.sp.encode(text, out_type=int) for text in texts]
    
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
        
        return [self.sp.decode(token_ids) for token_ids in batch_token_ids]
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size"""
        return self.sp.get_piece_size()
    
    def get_vocab(self) -> dict:
        """Get the vocabulary as a dictionary {token: id}"""
        vocab = {}
        for i in range(self.get_vocab_size()):
            vocab[self.sp.id_to_piece(i)] = i
        return vocab
    
    def save(self, path: str):
        """Save the tokenizer to a file"""
        # SentencePiece models are saved differently, this is just for compatibility
        # The actual sentencepiece model should already be saved to path
        pass
    
    @classmethod
    def from_pretrained(cls, path: str):
        """Load tokenizer from a pre-trained file"""
        return cls(model_path=path)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens to a single string
        
        Args:
            tokens: List of tokens to join
            
        Returns:
            Joined string
        """
        return " ".join(tokens)


def train_sentencepiece_tokenizer(data_paths: List[str], 
                                model_path: str,
                                vocab_size: int = 32000,
                                model_type: str = "bpe",  # Can be 'bpe', 'unigram', 'char', 'word'
                                pad_id: int = 0,
                                unk_id: int = 1,
                                bos_id: int = 2,
                                eos_id: int = 3) -> SentencePieceTokenizer:
    """
    Train a SentencePiece tokenizer on the given data
    
    Args:
        data_paths: List of paths to training data files
        model_path: Path to save the trained tokenizer (without extension)
        vocab_size: Size of the vocabulary to create
        model_type: Type of model ('bpe', 'unigram', 'char', 'word')
        pad_id: ID for padding token
        unk_id: ID for unknown token
        bos_id: ID for beginning-of-sentence token
        eos_id: ID for end-of-sentence token
        
    Returns:
        SentencePieceTokenizer instance
    """
    # Create a single text file for training by combining all data files
    combined_text_path = model_path + "_training_data.txt"
    
    with open(combined_text_path, 'w', encoding='utf-8') as combined_file:
        for data_path in data_paths:
            with open(data_path, 'r', encoding='utf-8') as data_file:
                combined_file.write(data_file.read())
                combined_file.write("\n")  # Add separator between files
    
    # Train the sentencepiece model
    spm.SentencePieceTrainer.train(
        input=combined_text_path,
        model_prefix=model_path,
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
        # Additional training parameters
        max_sentence_length=4096,
        character_coverage=1.0,  # Full coverage for all languages
        add_dummy_prefix=False,  # Don't add dummy prefix
        remove_extra_whitespaces=True,  # Clean up extra whitespaces
        split_digits=True,  # Split all digits
        split_by_whitespace=True,  # Split by whitespace
        normalization_rule_name="nmt_nfkc",  # Normalize using NMT NFKC rules
    )
    
    # Remove the temporary combined file
    if os.path.exists(combined_text_path):
        os.remove(combined_text_path)
    
    # Return the trained tokenizer
    return SentencePieceTokenizer(model_path=model_path + ".model")


def create_default_sentencepiece_tokenizer() -> SentencePieceTokenizer:
    """
    Create a default SentencePiece tokenizer with a basic setup
    This is mainly for compatibility when a tokenizer hasn't been trained yet
    """
    # This is a placeholder - in practice you'd train a tokenizer first
    raise NotImplementedError("Use train_sentencepiece_tokenizer instead to create a trained tokenizer")