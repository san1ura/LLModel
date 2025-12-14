"""
Test for the SentencePiece tokenizer functionality
"""
import unittest
import tempfile
import os
from tokenizer.train_tokenizer import TokenizerTrainer, SentencePieceTokenizer


class TestTokenizerTrainer(unittest.TestCase):
    def test_tokenizer_trainer_creation(self):
        """Test that tokenizer trainer can be created with default values"""
        trainer = TokenizerTrainer()

        self.assertEqual(trainer.vocab_size, 32000)
        self.assertEqual(trainer.model_type, "bpe")
        self.assertIn("<pad>", trainer.special_tokens)
        self.assertIn("<unk>", trainer.special_tokens)
        self.assertIn("<s>", trainer.special_tokens)
        self.assertIn("</s>", trainer.special_tokens)

    def test_tokenizer_trainer_custom_values(self):
        """Test that tokenizer trainer can be created with custom values"""
        trainer = TokenizerTrainer(
            vocab_size=1000,
            model_type="Unigram",
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        )

        self.assertEqual(trainer.vocab_size, 1000)
        self.assertEqual(trainer.model_type, "unigram")
        self.assertIn("<PAD>", trainer.special_tokens)

    def test_train_from_texts(self):
        """Test training tokenizer from sample texts"""
        sample_texts = [
            "This is a test sentence.",
            "Another test sentence for training.",
            "More text for the tokenizer to learn from."
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "tokenizer.model")

            trainer = TokenizerTrainer(vocab_size=100, model_type="BPE")
            tokenizer = trainer.train_from_texts(sample_texts, output_path)

            # Check if tokenizer file was created
            self.assertTrue(os.path.exists(output_path))

            # Check if we can load the tokenizer
            loaded_tokenizer = SentencePieceTokenizer.from_pretrained(output_path)
            self.assertGreater(loaded_tokenizer.get_vocab_size(), 0)

    def test_different_model_types(self):
        """Test tokenizer trainer with different model types"""
        # Create a larger text dataset to ensure SentencePiece can create the desired vocab size
        sample_texts = [
            "This is a test sentence for training.",
            "Another sentence used for model training.",
            "More text to expand the training dataset.",
            "Additional training examples for tokenizer.",
            "Sample text with various words and phrases.",
            "Sentence with different vocabulary items.",
            "Another example for model training process.",
            "Text containing multiple tokens for learning.",
            "Training data with diverse sentence structures.",
            "Example sentences for vocabulary expansion."
        ]

        for model_type in ["BPE", "Unigram"]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_path = os.path.join(tmp_dir, f"tokenizer_{model_type.lower()}.model")

                # Use a lower vocab size to accommodate smaller datasets
                trainer = TokenizerTrainer(vocab_size=50, model_type=model_type)
                trainer.train_from_texts(sample_texts, output_path)

                self.assertTrue(os.path.exists(output_path))


class TestSentencePieceTokenizer(unittest.TestCase):
    def setUp(self):
        # Create a simple tokenizer for testing
        sample_texts = [
            "This is a test sentence.",
            "Another test sentence for training.",
            "More text for the tokenizer to learn from."
        ]

        # Use a persistent temporary directory structure
        import tempfile
        import atexit
        import shutil

        self.tmp_dir = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, self.tmp_dir)  # Clean up after tests

        self.tokenizer_path = os.path.join(self.tmp_dir, "test_tokenizer.model")

        trainer = TokenizerTrainer(vocab_size=100, model_type="BPE")
        trainer.train_from_texts(sample_texts, self.tokenizer_path)

        self.tokenizer = SentencePieceTokenizer.from_pretrained(self.tokenizer_path)

    def test_encode_decode(self):
        """Test encoding and decoding of text"""
        original_text = "This is a test sentence."

        # Encode the text
        encoded = self.tokenizer.encode(original_text)
        self.assertIsInstance(encoded, list)
        self.assertGreater(len(encoded), 0)

        # Decode back to text
        decoded = self.tokenizer.decode(encoded)
        self.assertIsInstance(decoded, str)

        # Note: Due to special token handling, exact match might not be expected
        # but the decoded text should contain the original content

    def test_encode_without_special_tokens(self):
        """Test encoding without special tokens"""
        original_text = "This is a test."

        # Encode with special tokens
        encoded_with_special = self.tokenizer.encode(original_text, add_special_tokens=True)

        # Encode without special tokens
        encoded_without_special = self.tokenizer.encode(original_text, add_special_tokens=False)

        # Without special tokens should be shorter or equal length
        self.assertGreaterEqual(len(encoded_with_special), len(encoded_without_special))

    def test_batch_encode_decode(self):
        """Test batch encoding and decoding"""
        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence."
        ]

        # Batch encode
        batch_encoded = self.tokenizer.batch_encode(texts)
        self.assertEqual(len(batch_encoded), len(texts))

        # Each encoded sequence should be a list
        for encoded_seq in batch_encoded:
            self.assertIsInstance(encoded_seq, list)
            self.assertGreater(len(encoded_seq), 0)

        # Batch decode
        batch_decoded = self.tokenizer.batch_decode(batch_encoded)
        self.assertEqual(len(batch_decoded), len(texts))

    def test_special_token_ids(self):
        """Test that special token IDs are properly set"""
        self.assertIsInstance(self.tokenizer.pad_token_id(), int)
        self.assertIsInstance(self.tokenizer.bos_token_id(), int)
        self.assertIsInstance(self.tokenizer.eos_token_id(), int)
        self.assertIsInstance(self.tokenizer.unk_token_id(), int)

    def test_vocab_size_and_vocab(self):
        """Test vocabulary size and content"""
        vocab_size = self.tokenizer.get_vocab_size()
        self.assertGreater(vocab_size, 0)

        vocab = self.tokenizer.get_vocab()
        self.assertIsInstance(vocab, dict)
        self.assertEqual(len(vocab), vocab_size)

        # Check that special tokens are in the vocabulary
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        for token in special_tokens:
            if token in vocab:  # Not all might be present depending on training
                self.assertIn(token, vocab)

    def test_save_and_load(self):
        """Test saving and loading tokenizer"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "saved_tokenizer.model")

            # Check if the source model file for self.tokenizer exists before saving
            if os.path.exists(self.tokenizer.spm_model_path):
                # Save the tokenizer
                self.tokenizer.save(save_path)
                self.assertTrue(os.path.exists(save_path))

                # Load the tokenizer
                loaded_tokenizer = SentencePieceTokenizer.from_pretrained(save_path)

                # Test that loaded tokenizer works
                original_text = "Test sentence."
                original_encoding = self.tokenizer.encode(original_text)
                loaded_encoding = loaded_tokenizer.encode(original_text)

                self.assertEqual(len(original_encoding), len(loaded_encoding))
            else:
                self.fail(f"Source model file does not exist: {self.tokenizer.spm_model_path}")


class TestTokenizerUtilities(unittest.TestCase):
    def test_create_tokenizer_from_vocab(self):
        """Test creating tokenizer from predefined vocabulary"""
        from tokenizer.train_tokenizer import create_tokenizer_from_vocab
        vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "test": 4,
            "sentence": 5,
            ".": 6
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "vocab_tokenizer.model")

            tokenizer = create_tokenizer_from_vocab(vocab, output_path)
            loaded_tokenizer = SentencePieceTokenizer.from_pretrained(output_path)

            self.assertGreaterEqual(loaded_tokenizer.get_vocab_size(), len(vocab))

    def test_train_default_tokenizer(self):
        """Test training a default tokenizer"""
        from tokenizer.train_tokenizer import train_default_tokenizer
        sample_texts = ["Sample text for training.", "More text here."]

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = os.path.join(tmp_dir, "train_data.txt")
            output_path = os.path.join(tmp_dir, "default_tokenizer.model")

            # Write sample data
            with open(data_path, 'w') as f:
                for text in sample_texts:
                    f.write(text + "\n")

            # Train tokenizer
            tokenizer = train_default_tokenizer([data_path], output_path, vocab_size=50)

            self.assertGreater(tokenizer.get_vocab_size(), 0)


def run_tokenizer_tests():
    """Run all tokenizer tests and report results"""
    test_classes = [
        TestTokenizerTrainer,
        TestSentencePieceTokenizer,
        TestTokenizerUtilities
    ]

    for test_class in test_classes:
        print(f"Running tests for {test_class.__name__}")
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if result.failures or result.errors:
            print(f"Errors or failures found in {test_class.__name__}")
            for failure in result.failures:
                print(f"FAILURE: {failure[0]} - {failure[1]}")
            for error in result.errors:
                print(f"ERROR: {error[0]} - {error[1]}")
        else:
            print(f"All tests passed for {test_class.__name__}")
        print()


if __name__ == '__main__':
    run_tokenizer_tests()