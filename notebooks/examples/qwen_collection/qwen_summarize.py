from qwen_model import QwenChatInference, QwenVLInference
from pathlib import Path
import polars as pl
from tqdm import tqdm
from notebooks.examples.func import PipelineText


# path
path = Path('/media/kevin/data_4t/Test/search')
file = path / 'Description_Images_clean.parquet'
df = pl.read_parquet(file)
df = PipelineText().run(df)
print(df.shape, df['item_id'].n_unique())

# img
q = QwenVLInference()
lst_img = []
for i in tqdm(df['file_path'].to_list()):
    response = q.run(i)
    lst_img.append(response)
df = (
    df.with_columns(pl.Series(values=lst_img, name='img_info'))
    .with_columns(
        pl.concat_str([pl.col('item_name_clean'), pl.col('img_info')], separator=" ").alias('img_info')
    )
)

# text
q = QwenChatInference(flash_attention_2=False)
lst_text = []
for i in tqdm(df['description'].to_list()):
    response = q.run(i)
    lst_text.append(response)

lst_combine = [f'{_} {i} {t}' for _, i, t in zip(df['item_name'], lst_img, lst_text)]

df = (
    df.with_columns(pl.Series(values=lst_text, name='text_info'))
    .with_columns(
        pl.concat_str([pl.col('item_name_clean'), pl.col('text_info')], separator=" ").alias('text_info')
    )
    .with_columns(pl.Series(values=lst_combine, name='combine'))
)

# export
df.write_parquet(path / 'Description_Images_summary.parquet')
