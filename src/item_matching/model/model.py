import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoProcessor, SiglipVisionModel
from FlagEmbedding import BGEM3FlagModel
from accelerate.test_utils.testing import get_backend


class Model:
    def __init__(self, max_length: int = 50):
        self.device, _, _ = get_backend()
        self.max_length = max_length

        self.text_model = None
        self.img_processor = None
        self.img_model = None

    def get_img_model(self):
        """Get Image Model"""
        model_id = "google/siglip-base-patch16-224"
        self.img_processor = AutoProcessor.from_pretrained(model_id)

        config = {
            "pretrained_model_name_or_path": model_id,
            "torch_dtype": torch.bfloat16,
        }
        self.img_model = SiglipVisionModel.from_pretrained(**config).to(self.device)
        self.img_model = torch.compile(self.img_model)

    def get_text_model(self):
        model_id = "BAAI/bge-m3"
        self.text_model = BGEM3FlagModel(model_id, use_fp16=True)

    def process_text(self, list_text: list[str]):
        embeddings = self.text_model.encode(
            list_text,
            batch_size=512,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"]
        return embeddings

    def process_image(self, batch, col: str):
        images = [Image.open(i).convert("RGB") for i in batch[col]]
        inputs = self.img_processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.img_model(**inputs)
        pooled_output = outputs.pooler_output
        return {"image_embed": pooled_output}

    @staticmethod
    def pp_normalize(batch, col: str = "image_embed") -> dict:
        norm_embed = F.normalize(batch[col], p=2, dim=1).cpu().numpy()
        return {col: norm_embed}
