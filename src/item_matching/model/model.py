import torch
from loguru import logger
from PIL import Image
from torch.nn import functional as F
from transformers import AutoProcessor, SiglipVisionModel
from FlagEmbedding import BGEM3FlagModel

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class Model:
    def __init__(self):
        self.device = device
        self.text_model = None
        self.img_processor = None
        self.img_model = None
        logger.info(f'[Model] Run on: {self.device}')

    def get_img_model(self):
        """Get Image Model"""
        model_id = 'google/siglip-base-patch16-224'
        self.img_processor = AutoProcessor.from_pretrained(model_id)
        self.img_model = SiglipVisionModel.from_pretrained(model_id).to(self.device)

    def get_text_model(self):
        model_id = 'BAAI/bge-m3'
        self.text_model = BGEM3FlagModel(model_id, use_fp16=True)

    def process_text(self, list_text: list[str]):
        embeddings = self.text_model.encode(
            list_text,
            batch_size=512,
            max_length=80,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']
        return embeddings

    def process_image(self, batch, col: str):
        images = [Image.open(i).convert('RGB') for i in batch[col]]
        inputs = self.img_processor(images=images, return_tensors='pt').to(device)
        with torch.inference_mode():
            outputs = self.img_model(**inputs)
        pooled_output = outputs.pooler_output
        return {'img_embed': pooled_output}

    @staticmethod
    def pp_img(batch, model, processor, col: str) -> dict:
        images = [Image.open(i).convert('RGB') for i in batch[col]]
        inputs = processor(images=images, return_tensors='pt').to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        pooled_output = outputs.pooler_output
        embeddings = F.normalize(pooled_output, p=2, dim=1).cpu().numpy()
        return {'img_embed': embeddings}

    @ staticmethod
    def pp_normalize(batch, col: str = 'dense_embed') -> dict:
        norm_embed = F.normalize(batch[col], p=2, dim=1).cpu().numpy()
        return {col: norm_embed}

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
