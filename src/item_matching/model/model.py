import torch
from PIL import Image
from rich import print
from torch.nn import functional as F
from transformers import AutoProcessor, SiglipVisionModel
from FlagEmbedding import BGEM3FlagModel

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class Model:
    def __init__(self, verbose: bool = False):
        self.device = device
        self.text_model = None
        self.img_processor = None
        self.img_model = None
        if verbose:
            print(f'[Model] Device: {self.device}')

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
        return {'image_embed': pooled_output}

    @staticmethod
    def pp_normalize(batch, col: str = 'image_embed') -> dict:
        norm_embed = F.normalize(batch[col], p=2, dim=1).cpu().numpy()
        return {col: norm_embed}
