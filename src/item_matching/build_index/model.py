import torch
from loguru import logger
import sys
from PIL import Image
from torch.nn import functional as F

logger.remove()
logger.add(sys.stdout, colorize=True, format='<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{function}</cyan> | <level>{message}</level>')
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
        logger.info(f'[Model] Run on: {self.device}')

    def get_img(self, model_id: str = 'openai/clip-vit-base-patch32'):
        from transformers import AutoProcessor, CLIPVisionModel

        self.img_processor = AutoProcessor.from_pretrained(model_id)
        self.img_model = CLIPVisionModel.from_pretrained(model_id).to(self.device)
        logger.info(f'Image model: {model_id}')
        return self.img_model, self.img_processor

    @staticmethod
    def pp_sparse_tfidf(batch, vectorizer, col: str) -> dict:
        embeddings = vectorizer.transform(batch[col]).toarray()
        return {'tfidf_embed': embeddings}

    @staticmethod
    def pp_img(batch, col, processor, model):
        images = [Image.open(i).convert('RGB') for i in batch[col]]
        inputs = processor(images=images, return_tensors='pt').to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        pooled_output = outputs.pooler_output
        embeddings = F.normalize(pooled_output, p=2, dim=1).cpu().numpy()
        return {'img_embed': embeddings}
