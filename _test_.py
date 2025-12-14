{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "L137_VT00L2m"
   },
   "source": [
    "# Transformer Model Fine-tuning Notebook\n",
    "\n",
    "This notebook provides a comprehensive guide for fine-tuning the transformer model, including model loading, training, evaluation, and testing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "i2k2N51f0L2o"
   },
   "outputs": [],
   "source": [
    "# Install required packages (if not already installed)\n",
    "import sys\n",
    "import os\n",
    "\n",
    "# Add the project root to Python path\n",
    "project_root = \".\"\n",
    "sys.path.insert(0, project_root)\n",
    "\n",
    "# Import required packages\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "import numpy as np\n",
    "import json\n",
    "from tqdm import tqdm\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Import model components\n",
    "from model.transformer import Transformer, Config\n",
    "from tokenizer.train_tokenizer import SentencePieceTokenizer\n",
    "from training.train import Trainer\n",
    "from evaluation.benchmarks.model_eval import ModelEvaluator\n",
    "from serving.inference_opt.generate import InferenceEngine"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "R8qX5u7y0L2p"
   },
   "source": [
    "## 1. Model and Tokenizer Setup\n",
    "\n",
    "Load the pre-trained model and tokenizer. You can specify your own paths or create a new model from config."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "7yjQ80uI0L2p"
   },
   "outputs": [],
   "source": [
    "# Define model configuration\n",
    "config_path = \"config/default_config.json\"  # Path to your config file\n",
    "tokenizer_path = \"weights/tokenizer.model\"  # Path to your tokenizer\n",
    "model_path = \"checkpoints/model.pt\"  # Path to your pre-trained model (optional)\n",
    "\n",
    "# Load configuration\n",
    "try:\n",
    "    config = Config.load(config_path)\n",
    "    print(f\"Loaded configuration from {config_path}\")\n",
    "except FileNotFoundError:\n",
    "    print(f\"Config file not found at {config_path}, creating default config\")\n",
    "    config = Config(\n",
    "        vocab_size=32000,\n",
    "        d_model=512,\n",
    "        n_layers=8,\n",
    "        max_len=1024,\n",
    "        n_heads=8,\n",
    "        d_ff=1024,\n",
    "        dropout=0.1,\n",
    "        use_rope=True,\n",
    "        pos_type='rope',\n",
    "        attention_type='standard',\n",
    "        use_gradient_checkpointing=False,\n",
    "        norm_first=True,\n",
    "        initializer_range=0.02\n",
    "    )\n",
    "    config.save(config_path)\n",
    "\n",
    "# Load tokenizer\n",
    "try:\n",
    "    tokenizer = SentencePieceTokenizer.from_pretrained(tokenizer_path)\n",
    "    print(f\"Loaded tokenizer from {tokenizer_path}\")\n",
    "except FileNotFoundError:\n",
    "    print(f\"Tokenizer not found at {tokenizer_path}, please train or download one\")\n",
    "    # Example of training a tokenizer (this would need training data)\n",
    "    # tokenizer = train_default_tokenizer([\"data/train.txt\"], tokenizer_path)\n",
    "\n",
    "# Initialize model\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"Using device: {device}\")\n",
    "\n",
    "model = Transformer(config)\n",
    "model.to(device)\n",
    "print(f\"Initialized model with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters\")\n",
    "\n",
    "# Load pre-trained weights if available\n",
    "if os.path.exists(model_path):\n",
    "    model.load_state_dict(torch.load(model_path, map_location=device))\n",
    "    print(f\"Loaded model weights from {model_path}\")\n",
    "else:\n",
    "    print(f\"No pre-trained weights found at {model_path}, using randomly initialized model\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "D2qOy31a0L2q"
   },
   "source": [
    "## 2. Sample Data Preparation\n",
    "\n",
    "Create sample data for fine-tuning. In practice, you would load your actual dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "RQbJh22C0L2q"
   },
   "outputs": [],
   "source": [
    "class SampleDataset(Dataset):\n",
    "    def __init__(self, texts, tokenizer, max_length=512):\n",
    "        self.texts = texts\n",
    "        self.tokenizer = tokenizer\n",
    "        self.max_length = max_length\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.texts)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        text = self.texts[idx]\n",
    "        encoding = self.tokenizer.encode(text, add_special_tokens=True)\n",
    "        \n",
    "        # Truncate or pad to max_length\n",
    "        if len(encoding) > self.max_length:\n",
    "            encoding = encoding[:self.max_length]\n",
    "        else:\n",
    "            pad_token_id = self.tokenizer.pad_token_id\n",
    "            encoding = encoding + [pad_token_id] * (self.max_length - len(encoding))\n",
    "        \n",
    "        input_ids = torch.tensor(encoding, dtype=torch.long)\n",
    "        labels = input_ids.clone()\n",
    "        \n",
    "        return {\"input_ids\": input_ids, \"labels\": labels}\n",
    "\n",
    "# Sample text data for fine-tuning\n",
    "sample_texts = [\n",
    "    \"The quick brown fox jumps over the lazy dog. This is a sample sentence for fine-tuning.\",\n",
    "    \"Machine learning is a subset of artificial intelligence that focuses on algorithms.\",\n",
    "    \"Natural language processing allows computers to understand human language.\",\n",
    "    \"Deep learning models are based on neural networks with multiple layers.\",\n",
    "    \"Transformers are the state-of-the-art in natural language processing.\",\n",
    "    \"Fine-tuning pre-trained models is an efficient way to adapt them to specific tasks.\",\n",
    "    \"The attention mechanism helps models focus on relevant parts of the input.\",\n",
    "    \"PyTorch is a popular deep learning framework with dynamic computation graphs.\",\n",
    "    \"Reinforcement learning involves an agent learning to take actions in an environment.\",\n",
    "    \"Data preprocessing is a crucial step in machine learning pipelines.\"\n",
    "]\n",
    "\n",
    "# Create dataset\n",
    "dataset = SampleDataset(sample_texts, tokenizer)\n",
    "dataloader = DataLoader(dataset, batch_size=2, shuffle=True)\n",
    "\n",
    "print(f\"Created dataset with {len(dataset)} samples\")\n",
    "print(f\"Sample batch shape: {next(iter(dataloader))['input_ids'].shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0t7Q83860L2r"
   },
   "source": [
    "## 3. Fine-tuning Configuration\n",
    "\n",
    "Set up training parameters for fine-tuning."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "5X6r8Y6R0L2r"
   },
   "outputs": [],
   "source": [
    "# Fine-tuning parameters\n",
    "learning_rate = 5e-5\n",
    "epochs = 3\n",
    "warmup_steps = 10\n",
    "weight_decay = 0.01\n",
    "gradient_accumulation_steps = 2\n",
    "max_grad_norm = 1.0\n",
    "\n",
    "print(f\"Learning rate: {learning_rate}\")\n",
    "print(f\"Epochs: {epochs}\")\n",
    "print(f\"Warmup steps: {warmup_steps}\")\n",
    "print(f\"Weight decay: {weight_decay}\")\n",
    "print(f\"Gradient accumulation steps: {gradient_accumulation_steps}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "j4L4uZzv0L2r"
   },
   "source": [
    "## 4. Fine-tuning the Model\n",
    "\n",
    "Run the fine-tuning process."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "0Xy2178t0L2s"
   },
   "outputs": [],
   "source": [
    "# Move model to training mode\n",
    "model.train()\n",
    "\n",
    "# Set up optimizer\n",
    "optimizer = torch.optim.AdamW(\n",
    "    model.parameters(), \n",
    "    lr=learning_rate,\n",
    "    weight_decay=weight_decay\n",
    ")\n",
    "\n",
    "# Set up learning rate scheduler\n",
    "from transformers import get_linear_schedule_with_warmup\n",
    "\n",
    "total_steps = len(dataloader) * epochs\n",
    "scheduler = get_linear_schedule_with_warmup(\n",
    "    optimizer,\n",
    "    num_warmup_steps=warmup_steps,\n",
    "    num_training_steps=total_steps\n",
    ")\n",
    "\n",
    "# Training loop\n",
    "print(\"Starting fine-tuning...\")\n",
    "model.zero_grad()\n",
    "\n",
    "for epoch in range(epochs):\n",
    "    print(f\"\\nEpoch {epoch + 1}/{epochs}\")\n",
    "    total_loss = 0\n",
    "    \n",
    "    for step, batch in enumerate(tqdm(dataloader, desc=f\"Training Epoch {epoch + 1}\")):\n",
    "        input_ids = batch[\"input_ids\"].to(device)\n",
    "        labels = batch[\"labels\"].to(device)\n",
    "        \n",
    "        # Forward pass\n",
    "        outputs = model(input_ids, mask=None)  # For simplicity, not using attention mask\n",
    "        logits, _ = outputs  # Extract logits from the output tuple\n",
    "        \n",
    "        # Calculate loss\n",
    "        loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)\n",
    "        shift_logits = logits[..., :-1, :].contiguous()\n",
    "        shift_labels = labels[..., 1:].contiguous()\n",
    "        \n",
    "        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), \n",
    "                       shift_labels.view(-1))\n",
    "        \n",
    "        # Normalize loss\n",
    "        loss = loss / gradient_accumulation_steps\n",
    "        \n",
    "        # Backward pass\n",
    "        loss.backward()\n",
    "        \n",
    "        total_loss += loss.item()\n",
    "        \n",
    "        # Update weights after gradient accumulation steps\n",
    "        if (step + 1) % gradient_accumulation_steps == 0:\n",
    "            # Clip gradients\n",
    "            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)\n",
    "            \n",
    "            # Update weights\n",
    "            optimizer.step()\n",
    "            scheduler.step()\n",
    "            \n",
    "            # Clear gradients\n",
    "            optimizer.zero_grad()\n",
    "    \n",
    "    avg_loss = total_loss / len(dataloader)\n",
    "    print(f\"Average Loss for Epoch {epoch + 1}: {avg_loss:.4f}\")\n",
    "\n",
    "print(\"\\nFine-tuning completed!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "b0kZ38tj0L2s"
   },
   "source": [
    "## 5. Model Evaluation\n",
    "\n",
    "Evaluate the fine-tuned model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "y47k3L3L0L2s"
   },
   "outputs": [],
   "source": [
    "# Move model to evaluation mode\n",
    "model.eval()\n",
    "\n",
    "# Create an evaluator\n",
    "evaluator = ModelEvaluator(model, tokenizer, config)\n",
    "\n",
    "# Example evaluation (using a simple approach since we don't have a proper eval dataset)\n",
    "test_texts = [\n",
    "    \"The transformer model is\",\n",
    "    \"Fine-tuning allows us to adapt\", \n",
    "    \"Machine learning has revolutionized\",\n",
    "    \"Natural language processing involves\"\n",
    "]\n",
    "\n",
    "print(\"Sample Evaluations:\")\n",
    "for text in test_texts:\n",
    "    print(f\"\\nInput: {text}\")\n",
    "    \n",
    "    # Encode input\n",
    "    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0).to(device)\n",
    "    \n",
    "    # Generate output\n",
    "    with torch.no_grad():\n",
    "        generated = model.generate(\n",
    "            input_ids,\n",
    "            max_new_tokens=20,\n",
    "            temperature=0.7,\n",
    "            do_sample=True,\n",
    "            pad_token_id=tokenizer.pad_token_id\n",
    "        )\n",
    "    \n",
    "    # Decode output\n",
    "    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)\n",
    "    print(f\"Output: {output_text}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "f8yR9kWY0L2t"
   },
   "source": [
    "## 6. Inference Testing\n",
    "\n",
    "Test the fine-tuned model with the inference engine."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "37tBdRbB0L2t"
   },
   "outputs": [],
   "source": [
    "# Create inference engine\n",
    "inference_engine = InferenceEngine(model, tokenizer, config)\n",
    "\n",
    "# Test prompts\n",
    "test_prompts = [\n",
    "    \"The future of artificial intelligence is\",\n",
    "    \"In machine learning, a transformer model\",\n",
    "    \"Natural language processing has advanced significantly with\",\n",
    "    \"Fine-tuning a pre-trained model allows\"\n",
    "]\n",
    "\n",
    "# Generate responses\n",
    "print(\"Inference Tests:\")\n",
    "for i, prompt in enumerate(test_prompts):\n",
    "    print(f\"\\n{i+1}. Prompt: {prompt}\")\n",
    "    \n",
    "    # Generate response\n",
    "    response = inference_engine.generate(\n",
    "        inputs=prompt,\n",
    "        max_new_tokens=30,\n",
    "        temperature=0.8,\n",
    "        top_k=50,\n",
    "        top_p=0.95,\n",
    "        do_sample=True\n",
    "    )\n",
    "    \n",
    "    print(f\"Response: {response}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "qJXwYv490L2t"
   },
   "source": [
    "## 7. Save Fine-tuned Model\n",
    "\n",
    "Save the fine-tuned model weights."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "Rb34jZv40L2t"
   },
   "outputs": [],
   "source": [
    "# Define the path to save the fine-tuned model\n",
    "output_dir = \"checkpoints\"\n",
    "os.makedirs(output_dir, exist_ok=True)\n",
    "model_save_path = os.path.join(output_dir, \"fine_tuned_model.pt\")\n",
    "\n",
    "# Save the model\n",
    "model.save(model_save_path)\n",
    "print(f\"Fine-tuned model saved to {model_save_path}\")\n",
    "\n",
    "# Optionally, save the configuration as well\n",
    "config_save_path = os.path.join(output_dir, \"fine_tuned_config.json\")\n",
    "config.save(config_save_path)\n",
    "print(f\"Configuration saved to {config_save_path}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "o9j301mN0L2u"
   },
   "source": [
    "## 8. Performance Benchmarking\n",
    "\n",
    "Benchmark the fine-tuned model's performance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "7YdX9vQ90L2u"
   },
   "outputs": [],
   "source": [
    "# Benchmark inference performance\n",
    "benchmark_results = inference_engine.benchmark_inference(\n",
    "    input_text=\"Performance benchmark test: \",\n",
    "    num_generations=5,\n",
    "    max_new_tokens=20\n",
    ")\n",
    "\n",
    "print(\"Performance Benchmark Results:\")\n",
    "for metric, value in benchmark_results.items():\n",
    "    print(f\"{metric}: {value:.4f}\" if isinstance(value, float) else f\"{metric}: {value}\")\n",
    "\n",
    "# Visualize benchmark results\n",
    "plt.figure(figsize=(12, 4))\n",
    "\n",
    "# Plot generation time\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.bar(['Avg Time'], [benchmark_results['avg_generation_time']])\n",
    "plt.ylabel('Time (seconds)')\n",
    "plt.title('Average Generation Time')\n",
    "\n",
    "# Plot tokens per second\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.bar(['Tokens/sec'], [benchmark_results['tokens_per_second']])\n",
    "plt.ylabel('Tokens per second')\n",
    "plt.title('Generation Throughput')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "qX42z10D0L2u"
   },
   "source": [
    "## 9. Fine-tuning with Different Strategies\n",
    "\n",
    "Try different fine-tuning approaches like LoRA."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "e3209X8B0L2u"
   },
   "outputs": [],
   "source": [
    "# This would be a more advanced implementation using LoRA\n",
    "# For now, we'll just demonstrate the concept with a simple parameter freezing approach\n",
    "\n",
    "# Freeze some layers for parameter-efficient fine-tuning\n",
    "def freeze_layers(model, num_layers_to_freeze=4):\n",
    "    \"\"\"Freeze the first num_layers_to_freeze transformer blocks\"\"\"\n",
    "    for i, layer in enumerate(model.layers):\n",
    "        if i < num_layers_to_freeze:\n",
    "            for param in layer.parameters():\n",
    "                param.requires_grad = False\n",
    "        else:\n",
    "            for param in layer.parameters():\n",
    "                param.requires_grad = True\n",
    "\n",
    "# Apply freezing (only for the first 4 layers)\n",
    "freeze_layers(model, num_layers_to_freeze=4)\n",
    "\n",
    "# Count trainable parameters\n",
    "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
    "total_params = sum(p.numel() for p in model.parameters())\n",
    "\n",
    "print(f\"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}% of total)\")\n",
    "print(f\"Total parameters: {total_params:,}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "z75t6Y1C0L2u"
   },
   "source": [
    "## 10. Conclusion\n",
    "\n",
    "This notebook demonstrated how to fine-tune a transformer model, evaluate it, and run inference. We covered:\n",
    "\n",
    "1. Loading the pre-trained model and tokenizer\n",
    "2. Preparing sample data for fine-tuning\n",
    "3. Setting up fine-tuning parameters\n",
    "4. Running the fine-tuning process\n",
    "5. Evaluating the fine-tuned model\n",
    "6. Testing inference with the fine-tuned model\n",
    "7. Saving the fine-tuned model\n",
    "8. Benchmarking performance\n",
    "9. Exploring parameter-efficient fine-tuning\n",
    "\n",
    "In practice, you would replace the sample data with your actual fine-tuning dataset and adjust the hyperparameters based on your specific use case."
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
