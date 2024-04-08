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
        logger.info(f'[Model] Run on: {self.device}')

    def get_img_model(self, model_id: str = 'openai/clip-vit-base-patch32'):
        from transformers import AutoProcessor, CLIPVisionModel

        img_processor = AutoProcessor.from_pretrained(model_id)
        img_model = CLIPVisionModel.from_pretrained(model_id).to(self.device)
        logger.info(f'Image model: {model_id}')
        return img_model, img_processor

    def get_text_model(self, model_id: str = 'BAAI/bge-m3'):
        from FlagEmbedding import BGEM3FlagModel

        model = BGEM3FlagModel(model_id, use_fp16=True)
        return model

    @staticmethod
    def pp_sparse_tfidf(batch, vectorizer, col: str) -> dict:
        embeddings = vectorizer.transform(batch[col]).toarray()
        return {'tfidf_embed': embeddings}

    @staticmethod
    def pp_img(batch, model, processor, col: str) -> dict:
        images = [Image.open(i).convert('RGB') for i in batch[col]]
        inputs = processor(images=images, return_tensors='pt').to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        pooled_output = outputs.pooler_output
        embeddings = F.normalize(pooled_output, p=2, dim=1).cpu().numpy()
        return {'img_embed': embeddings}

    @staticmethod
    def pp_dense(batch, model, col: str) -> dict:
        embeddings = model.encode(
            batch[col],
            batch_size=512,
            max_length=80,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']
        embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=1).cpu().numpy()
        return {'dense_embed': embeddings}
