from qwen_model import QwenChatInference, QwenVLInference
from pathlib import Path
import polars as pl
from tqdm import tqdm


# path
path = Path('/media/kevin/data_4t/Test')
file = path / 'Description_Images_clean.parquet'
df = pl.read_parquet(file).head(200)
print(df.shape, df['item_id'].n_unique())

# img
q = QwenVLInference()
lst_img = []
for i in tqdm(df['file_path'].to_list()):
    response = q.run(i)
    lst_img.append(response)
df = df.with_columns(pl.Series(values=lst_img, name='img_info'))

# text
q = QwenChatInference(flash_attention_2=False)
lst_text = []
for i in tqdm(df['description'].to_list()):
    response = q.run(i)
    lst_text.append(response)
df = df.with_columns(pl.Series(values=lst_img, name='text_info'))

# export
df.write_parquet(path / 'Description_Images_summary.parquet')
