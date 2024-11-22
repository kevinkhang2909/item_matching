import duckdb
import numpy as np
from core_pro.ultilities import make_sync_folder, create_batch_index
from src.item_matching.func.function_image import PipelineImage
from src.item_matching.func.function_text import PipelineText
from datasets import Dataset, concatenate_datasets
from src.item_matching.model.model import Model


# path
path = make_sync_folder('Item_Matching_Test')
file = path / 'data_sample_ELHA_0.parquet'

query = f"""select * from read_parquet('{file}')"""
df = duckdb.sql(query).pl()

df = PipelineText().run(df)
# df, df_img = PipelineImage(path).run(df)
df = df.unique(['item_id'])
# df.write_parquet(path / 'clean.parquet')

MATCH_BY = 'text'
model = Model()
if MATCH_BY == 'text':
    model.get_text_model()
    col_embed = 'text_embed'
else:
    model.get_img_model()
    col_embed = 'image_embed'

run = create_batch_index(df.shape[0], 10_000)
num_chunks = len(run)
for i, idx in run.items():
    # Check if exists:
    dataset_name = path / 'ds' / f'{i}'
    array_name = path / 'array' / f'{i}.npy'
    if dataset_name.exists():
        continue

    # Load Chunk
    start_idx, end_idx = idx[0], idx[-1]
    dataset_chunk = Dataset.from_polars(df[start_idx:end_idx])
    print(f'Shard [{i}/{num_chunks}]: start {start_idx:,.0f} end {end_idx:,.0f}')

    # Process dataset
    if MATCH_BY == 'text':
        embeddings = model.process_text(dataset_chunk[col_embed])
        dset_embed = Dataset.from_dict({col_embed: embeddings})
        dataset_chunk = concatenate_datasets([dataset_chunk, dset_embed], axis=1)
    else:
        dataset_chunk = dataset_chunk.map(
            model.process_image,
            batch_size=512,
            batched=True,
            fn_kwargs={'col': col_embed}
        )

    # Normalize
    dataset_chunk.set_format(type='torch', columns=[col_embed], output_all_columns=True)
    if MATCH_BY == 'image':
        dataset_chunk = dataset_chunk.map(Model.pp_normalize, batched=True, fn_kwargs={'col': col_embed})
    dataset_chunk.set_format(type='numpy', columns=[col_embed], output_all_columns=True)

    # Save chunk
    np.save(array_name, dataset_chunk[col_embed])
    dataset_chunk.save_to_disk(str(dataset_name))
