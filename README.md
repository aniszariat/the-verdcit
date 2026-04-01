# 🧠 The Verdict — GPT-style LLM from Scratch

This project demonstrates the end-to-end implementation of a **Large Language Model (LLM)** inspired by the book *Build a Large Language Model (from Scratch)* by Sebastian Raschka.

The goal is to understand how modern generative AI systems work by building a **GPT-style transformer model from scratch**, training it, and adapting it to specific tasks.

---

## 🚀 Features

- Implementation of a **Transformer (GPT-style) architecture**
- Custom **self-attention mechanism**
- Text **tokenization and preprocessing**
- Model **pretraining on text data**
- **Fine-tuning** on specific tasks
- Text **generation (inference)**
- Lightweight and runnable on a local machine

---

## 🏗️ Project Structure

```
theVerdict/
├── chapter02_tokenization/       # Text tokenization & preprocessing
│   ├── tokenizer.py              # Tokenizer implementation
│   ├── theVerdict.py             # Main tokenization pipeline
│   ├── dataLoaderProcess.py      # Data loading utilities
│   ├── split.py                  # Train/validation split
│   ├── torchTest.py              # PyTorch tests
│   ├── tiktokenTest.py           # TikToken tests
│   └── the-verdict.txt           # Training dataset
│
├── chapter03_attentionMechanism/ # Self-attention mechanisms
│   ├── selfAttentionClass.py     # Basic self-attention
│   ├── causalAttentionClass.py   # Causal attention (GPT-style)
│   ├── multiHeadAttentionClass.py # Multi-head attention
│   ├── attention-mechanism-implementation.py
│   ├── attention-mechanisms.py
│   ├── causal-attention.py
│   ├── softmax.py
│   └── dropout.py
│
├── chapter04_GPTModel/           # GPT transformer model
│   ├── GPTModelClass.py          # Main GPT model class
│   ├── DummyGPTModelClass.py     # Simplified model variant
│   ├── transformerBlockClass.py  # Transformer block implementation
│   ├── multiHeadAttentionClass.py
│   ├── feedForwardNetwork.py     # Feed-forward network
│   ├── layerNormalizationClass.py # Layer normalization
│   ├── activationFunction.py     # Activation functions
│   ├── shortcutConnections.py    # Skip connections
│   ├── gpt_config.py             # Model configuration
│   ├── gpt.py                    # GPT utilities
│   ├── plots.py                  # Visualization tools
│   └── tb.py                     # Tensorboard utilities
│
├── README.md
└── .gitignore
```


---

## ⚙️ How It Works

### 1. Data Preparation
- Clean and preprocess raw text
- Convert text into tokens
- Build input-target sequences for training

### 2. Transformer Model
- Token embeddings + positional encoding
- Multi-head self-attention
- Feed-forward neural networks
- Stacked transformer blocks

### 3. Pretraining
- Train on a general corpus
- Objective: **next-token prediction**
- Optimization via backpropagation

### 4. Fine-Tuning
- Adapt model to specific tasks:
  - Text classification
  - Domain-specific generation

### 5. Inference
- Generate text from prompts
- Control output using sampling strategies (e.g., temperature)

---

## 🧪 Usage

### Install dependencies
```bash
pip install -r requirements.txt