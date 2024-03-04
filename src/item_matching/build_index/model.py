import torch
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, colorize=True, format='<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}:{function}</cyan> | <level>{message}</level>')
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class Model:
    def __init__(self):
        self.device = device
        self.dense_model = None
        self.dense_tokenizer = None
        self.sparse_tokenizer = None
        self.sparse_model = None
        self.img_model = None
        self.img_processor = None
        logger.info(f'Run model on device: {self.device}')

    @staticmethod
    def pp_sparse_tfidf(batch, vectorizer, col: str) -> dict:
        embeddings = vectorizer.transform(batch[col]).toarray()
        return {'sparse_embed': embeddings}
