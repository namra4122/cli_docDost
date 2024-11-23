import warnings
import logging
from pathlib import Path
from docdost import (
    Config,
    ModelManager,
    DocumentProcessor,
    RedisManager
)
from docdost.utils.helpers import setup_logging
import pandas as pd
from halo import Halo
from rich import print
warnings.filterwarnings("ignore")

def main():
    logger = logging.getLogger(__name__)
    setup_logging(log_level="ERROR")

    try:
        logger.info("---------------------DocDost Started---------------------")
        
        with Halo(text='Just a moment... the models are flexing their neural muscles 💪🤖',color='blue',text_color='green', spinner='dots'):
            config = Config()
            doc_processor = DocumentProcessor(config)
            model_manager = ModelManager(config)
            redis_manager = RedisManager(config)

        # Get PDF path from user
        print("[green]Drop your PDF path here... we promise not to judge your folder names 😅📂[/green]")
        pdf_path = Path(input())
        
        # Process document
        with Halo(text='Hold tight! Your PDF is getting the VIP treatment 🏆📄',color='blue',text_color='green', spinner='dots'):
            page_data = doc_processor.load_pdf(pdf_path)
            chunk_data = doc_processor.chunk_text(page_data)
            chunks = model_manager.embed_chunks(chunk_data)
            redis_data = pd.DataFrame(chunks).to_dict(orient="records")
            redis_manager.add_to_redis(redis_data)
        
        print("[green]Shoot your question! Your PDF is ready to spill the beans (or bytes) 🤔💾[/green]")
        query = input()
        with Halo(text='Hold on, your answer is being handcrafted by digital ninjas 🥷✨',color='blue',text_color='green', spinner='dots'):
            search_result = redis_manager.similarity_search(
                query = query,
                embedding_model = model_manager.embedding_model,
                topk = 5
            )
            context_items = []
            for item in search_result:
                context_items.append(item['sentence_chunks'])
            tokenized_prompt = model_manager.format_promp(query=query,context=context_items)
            llm_response = model_manager.generate_response(tokenized_prompt=tokenized_prompt, max_new_tokens=256, temperature=0.5)
        
        print(llm_response)
    except Exception as e:
        print(f"[bold red]ERROR: {e}[/bold red]")

if __name__ == '__main__':
    main()