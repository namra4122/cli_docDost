from dataclasses import dataclass
import torch

@dataclass
class Config:
    embedding_model_name: str = "all-distilroberta-v1"
    # "sentence-transformers/all-mpnet-base-v2"
    # "all-distilroberta-v1"
    llm_model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    chunk_size: int = 12
    chunk_overlap: int = 2
    embedding_size: int = 768
    redis_url: str = "redis://localhost:6379"
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )