import logging
from pathlib import Path

def setup_logging(log_level: str = "ERROR") -> None:
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('docdost.log')
        ]
    )