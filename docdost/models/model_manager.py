import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BatchEncoding
from ..config import Config

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config: Config):
        self.config = config,
        self.embedding_model, self.tokenizer, self.llm_model = self._init_models()
    
    def _init_models(self) -> Tuple[SentenceTransformer,AutoTokenizer,AutoModelForCausalLM]:
        try:
            attn_implementation = "flash_attention_2" if (
                is_flash_attn_2_available() and 
                torch.cuda.get_device_capability(0)[0] >= 8
            ) else "sdpa"

            logger.info(f"Using attention implementation: {attn_implementation}")
            
            logger.info(f"Using model: {self.config[0].llm_model_name}")

            embedding_model = SentenceTransformer(
                self.config[0].embedding_model_name,
                device=self.config[0].device,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.config[0].llm_model_name,
                clean_up_tokeniztion_spaces = True
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.config[0].llm_model_name,
                torch_dtype = torch.float16,
                quantization_config = None,
                low_cpu_mem_usage = False,
                attn_implementation = attn_implementation,
            ).to(self.config[0].device)

            return embedding_model,tokenizer,model
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
            raise

    def embed_chunks(self,chunks:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        try:
            for chunk in chunks:
                embedding = self.embedding_model.encode(chunk["sentence_chunks"])
                chunk["byte_embeddings"] = np.array(embedding, dtype=np.float32).tobytes()
            return chunks
        except Exception as e:
            logging.error(f"Error embedding data chunks: {e}")
            raise
    
    def format_promp(self,query: str, context: List[str]) -> str:
        try:
            base_prompt = """Your task is to understand the user's query and provide an answer based on the most relevant context provided. If the context is not related to the query, repond with "I'm sorry, but I dont' have enough relevant information to answer that question accurately.

            Context:
            {context}

            User query: {query}
            Please provide a helpful and accurate response based on the given context:"""

            formatted_prompt = base_prompt.format(context=context,query=query)

            dialogue_template = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": formatted_prompt},
            ]

            text_prompt = self.tokenizer.apply_chat_template(
                conversation = dialogue_template,
                tokenize = False,
                add_generation_prompt = True
            )

            return self.tokenizer(text_prompt, return_tensors = "pt").to(self.config[0].device)
        except Exception as e:
            logging.error(f"Error in formatting prompt: {e}")
            raise
    
    
    def generate_response(self,tokenized_prompt: BatchEncoding, max_new_tokens: int = 256, temperature: float = 0.5) -> str:
        try:
            tokenized_response = self.llm_model.generate(**tokenized_prompt,
                                               temperature=temperature,
                                               do_sample=True,
                                               max_new_tokens = max_new_tokens)
            response = self.tokenizer.decode(tokenized_response[0])
            return response
        except Exception as e:
            logging.error(f"Error while llm generation: {e}")
            raise