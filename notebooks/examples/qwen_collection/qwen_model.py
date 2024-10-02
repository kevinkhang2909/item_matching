from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
from pathlib import Path
from time import perf_counter


class QwenVLInference:
    def __init__(
            self,
            min_pixels: int = 64 * 28 * 28,
            max_pixels: int = 512 * 28 * 28,
            flash_attention_2: bool = False,
    ):
        self.model_id = 'Qwen/Qwen2-VL-2B-Instruct'
        self.model = None
        self.processor = None
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.flash_attention_2 = flash_attention_2

        self._load_model()
        self.device = self.model.device
        print(
            f'[QwenVL Infer] '
            f'Device: {self.device}, Flash Attention 2: {self.flash_attention_2}'
        )

    def _load_model(self):
        config = {
            'pretrained_model_name_or_path': self.model_id,
            'torch_dtype': 'auto',
            'device_map': 'auto',
            'attn_implementation': None,
        }
        if self.flash_attention_2:
            config.update({
                'torch_dtype': torch.bfloat16,
                'attn_implementation': 'flash_attention_2'
            })
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(**config)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )

    def run(self, file_path: Path, verbose: bool = False):
        start = perf_counter()

        # Prompt
        promp_vl = """
        You are an AI assistant specialized in analyzing and summarizing product images from e-commerce sites. Your task is to extract key visual information from product images and provide a structured summary that highlights the most important visual aspects and details. The output should be formatted in JSON for easy integration with various systems.

        Instructions:
        1. Carefully analyze the provided product image.
        2. Identify and describe key visual elements, including but not limited to:
        - Product type
        - Shop name
        - Main colors and color scheme
        - Product shape and design
        - Visible features or components
        - Branding elements (if any)
        - Setting or context (if applicable)
        - Quality and style of the image (e.g., studio shot, lifestyle image, etc.)
        3. Estimate the number of items shown in the image.
        4. Identify any text visible in the image.
        5. Note any unique or standout visual aspects of the product.
        6. If multiple views of the product are provided, describe each briefly.
        7. Use clear, concise language that accurately describes what you see.
        8. Avoid making assumptions about features that aren't visibly apparent.
        # 9. Format the output as a JSON object

        Remember, your goal is to provide an accurate, objective description of the visual elements in the product image, formatted in a way that's easy to process programmatically.
        Focus on what you can actually see in the image, without making assumptions about technical specifications or features that aren't visually apparent.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(file_path),
                        "resized_height": 224,
                        "resized_width": 224,
                    },
                    {"type": "text", "text": promp_vl},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        )
        inputs = inputs.to(self.device)
        num_tokens = len(inputs)

        start_time = perf_counter()
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        elapsed_time = (perf_counter() - start_time) / 60
        tpm = num_tokens / elapsed_time

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        if verbose:
            print(
                f'[Qwen VL] \n'
                f'Time: {perf_counter() - start:,.0f}s \n'
                f'Prompt: {num_tokens} tokens, {tpm:,.2f} tokens-per-minute'
            )
        return output_text[0]


class QwenChatInference:
    def __init__(self, flash_attention_2: bool = False):
        self.model = None
        self.tokenizer = None
        self.flash_attention_2 = flash_attention_2
        self._load_model()
        self.device = self.model.device
        print(
            f'[QwenChat Infer] '
            f'Device: {self.device}, Flash Attention 2: {self.flash_attention_2}'
        )

    def _load_model(self):
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        config = {
            'pretrained_model_name_or_path': model_name,
            'torch_dtype': 'auto',
            'device_map': 'auto',
            'attn_implementation': None,
        }
        if self.flash_attention_2:
            config.update({
                'torch_dtype': torch.bfloat16,
                'attn_implementation': 'flash_attention_2'
            })

        self.model = AutoModelForCausalLM.from_pretrained(**config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run(self, description: str, verbose: bool = False):
        start = perf_counter()

        prompt = f"""
        You are an AI assistant specialized in creating concise and informative summaries of e-commerce product descriptions. Your task is to distill lengthy product descriptions into clear, engaging summaries that highlight the most important features and benefits for potential customers.

        Instructions:
        1. Read the full product description carefully.
        2. Identify the key features, specifications, and unique selling points of the product.
        3. Summarize the product, ensuring you capture the following elements:
        - Product name
        - Brief description (1-2 sentences)
        - Key features (3-5 bullet points)
        - Target audience or ideal use case (if mentioned)
        - Warranty
        - Specifications: Color, Size, Brand (Color must be in text not code)
        5. Use clear, concise language that's easy for customers to understand.
        6. Avoid using technical jargon unless it's essential to describing the product.
        7. Maintain a neutral tone, focusing on facts rather than marketing language.
        8. If there are multiple variations or models of the product, mention this briefly.
        9. Format the output as a JSON object

        Remember, your goal is to create a structured summary that quickly gives potential customers the most important information about the product, helping them make informed purchasing decisions, while providing the data in a format that's easy to process programmatically.
        Summarize the product description below, delimited by triple 
        backticks.

        Description: ```{description}```
        """

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        num_tokens = len(model_inputs)

        start_time = perf_counter()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256
        )
        elapsed_time = (perf_counter() - start_time) / 60
        tpm = num_tokens / elapsed_time

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if verbose:
            print(
                f'[Qwen Chat] \n'
                f'Time: {perf_counter() - start:,.0f}s \n'
                f'Prompt: {num_tokens} tokens, {tpm:,.2f} tokens-per-minute'
            )
        return response
