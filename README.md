# CLI DocDost

A powerful RAG (Retrieval-Augmented Generation) Chatbot that runs locally on your system, enabling intelligent document interaction and question-answering capabilities.

## 🚀 Features

- 📄 Advanced PDF document processing and analysis
- 🔄 Smart text chunking and semantic embedding generation
- 🔍 High-performance Redis-based vector search
- 🤖 LLM-powered intelligent responses to your queries
- 💻 Completely local execution for data privacy
- 🎯 Accurate context-aware answers

## 🛠️ Prerequisites

- Python 3.8 or higher
- Docker (for Redis Stack)
- Hugging Face account
- CUDA-compatible GPU (optional, for better performance)

## ⚡ Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/namra4122/cli_docDost.git
cd cli_docdost

# Create and activate virtual environment
python -m venv venv
# On Unix/macOS
source venv/bin/activate
# On Windows
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Hugging Face Setup

You'll need to authenticate with Hugging Face to download the required models:

```bash
huggingface-cli login
```

### 3. Redis Setup

Start the Redis Stack container:

```bash
docker run -d --name redis-stack \
    -p 6379:6379 \
    -p 8001:8001 \
    redis/redis-stack:latest

```

You can monitor your Redis instance at http://localhost:8001

## 💫 Usage

1. Start the application:

```bash
python main.py
```

2. Follow the interactive prompts to:

- Input your PDF document path
- Ask questions about your document
- Get AI-generated responses based on the document content

## 🔧 Configuration

The application can be configured through config.py:

- Embedding model selection
- LLM model parameters
- Chunking settings
- Redis connection details

## 🤝 Contributing

Contributions are warmly welcomed! Here's how you can help:

- Fork the repository
- Create your feature branch (`git checkout -b feature/AmazingFeature`)
- Commit your changes (`git commit -m 'Add some AmazingFeature'`)
- Push to the branch (`git push origin feature/AmazingFeature`)
- Open a Pull Request

## ⭐ Support

If you find this project helpful, please consider giving it a star on GitHub!
