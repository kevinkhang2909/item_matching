from datasets import concatenate_datasets, load_from_disk, Dataset
from pathlib import Path
import polars as pl
from re import search
import duckdb


path = Path('/media/kevin/75b198db-809a-4bd2-a97c-e52daa6b3a2d/item_match')
path_tmp = {
    'db': {'db_array': path / 'db_array', 'db_ds': path / 'db_ds'},
    'q': {'q_array': path / 'q_array', 'q_ds': path / 'q_ds'},
}

dataset_db = concatenate_datasets([
    load_from_disk(str(f)) for f in sorted(path_tmp['db']['db_ds'].glob('*'))
])
path_query = Path('/media/kevin/75b198db-809a-4bd2-a97c-e52daa6b3a2d/item_match/q_clean.parquet')
query = f"""
select * 
from read_parquet('{path_query}') 
"""
df_q = duckdb.sql(query).pl()


# Add index
# col_embed = 'dense_embed'
# path_index = path / 'index'
# dataset_db.load_faiss_index(col_embed, path_index / f'ip.index')
#
# # Dataset query shard
# dataset_q = concatenate_datasets([
#     load_from_disk(str(f)) for f in sorted(path_tmp['q']['q_ds'].glob('*'))
# ])

# Batch query
path_result = path / 'result'
# make_dir(self.path_result)

# query_batch_size = 50_000
# top_k = 100
# num_batches = len(dataset_q) // query_batch_size
# print(num_batches, query_batch_size)
# # logger.info(f'[Matching] Start retrieve: num batches {num_batches}')
# # start = perf_counter()
# for idx, i in enumerate(range(0, len(dataset_q), query_batch_size)):
#     # init
#     file_name_result = path_result / f'result_{idx}.parquet'
#     file_name_score = path_result / f'score_{idx}.parquet'
#     if file_name_result.exists():
#         continue
# 
#     # start
#     # start_batch = perf_counter()
#     if i + query_batch_size >= len(dataset_q):
#         batched_queries = dataset_q[i:]
#     else:
#         batched_queries = dataset_q[i:i + query_batch_size]
# 
#     # query
#     score, result = dataset_db.get_nearest_examples_batch(
#         col_embed,
#         batched_queries[col_embed],
#         k=top_k
#     )
#     # export
#     dict_ = {f'score_{col_embed}': [list(i) for i in score]}
#     df_score = pl.DataFrame(dict_)
#     df_score.write_parquet(file_name_score)
#     df_result = pl.DataFrame(result).drop([col_embed])
#     df_result.write_parquet(file_name_result)
# 
#     # log
#     # print(f"[Matching] Batch {idx}/{num_batches} match result shape: {df_result.shape} "
#     #       f"{perf_counter() - start_batch:,.2f}s")
#     del score, result, df_score, df_result

df_score = (
    pl.concat([pl.read_parquet(f) for f in sorted(path_result.glob('score*.parquet'))])
)
df_result = (
    pl.concat([pl.read_parquet(f) for f in sorted(path_result.glob('result*.parquet'))])
)
df_match = pl.concat([df_q, df_result, df_score], how='horizontal')
col_explode = [i for i in df_match.columns if search('db|score', i)]
df_match = df_match.explode(col_explode)
path_match_result = path / 'result_match'
export_type = 'csv'
cat = 'all'
file_name = path_match_result / f'{cat}.{export_type}'
df_match.write_csv(file_name)
