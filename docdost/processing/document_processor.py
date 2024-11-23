import logging
import re
import fitz
from pathlib import Path
from typing import List,Dict,Any
from spacy.lang.en import English
from ..config import Config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config,
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")

    @staticmethod
    def clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
    
    def load_pdf(self, path: Path) -> List[Dict[str,Any]]:
        if not path.exists():
            raise FileNotFoundError(f"File not found at path: {path}")

        pages_data = []
        try:
            with fitz.open(path) as doc:
                for page_num, page in enumerate(doc):
                    text = self.clean_text(page.get_text())
                    sentences = [str(sent) for sent in self.nlp(text).sents]
                    pages_data.append({
                        "page_number": page_num,
                        "sentences": sentences
                    })
            return pages_data
        except Exception as e:
            logger.error(f"Error loading a document: {e}")
            raise
    
    def chunk_text(self,page_data: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        def split_text(input_list: List[str], slice_size: int, overlap: int) -> List[List[str]]:
            return [
                input_list[max(0, i - overlap) : i + slice_size]
                for i in range(0, len(input_list), slice_size)
            ]
        
        chunk = []
        try:
            for items in page_data:
                sentence_chunks = split_text(items["sentences"], self.config[0].chunk_size, self.config[0].chunk_overlap)

                for sents in sentence_chunks:
                    chunk_text = " ".join(sents).strip()
                    chunk_text = re.sub(r"\.([A-Z])", r". \1", chunk_text)
                    chunk.append({
                        "page_number" : items["page_number"],
                        "sentence_chunks": chunk_text
                    })
            return chunk
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise
        