from pathlib import Path
import polars as pl
from autofaiss import build_index
from FlagEmbedding import BGEM3FlagModel
from datasets import Dataset
import numpy as np
from core_pro.ultilities import make_dir
from item_matching import PipelineText


def run(file_raw: str, col: str = 'q_item_name_clean'):
    path = Path('/media/kevin/data_4t/Test/search')
    file = path / f'{file_raw}.parquet'
    df = pl.read_parquet(file)
    print(df.shape, df['item_id'].n_unique())

    df_q = PipelineText(mode='q').run(df)

    path_tmp_array = path / 'tmp/array'
    make_dir(path_tmp_array)

    max_len_dict = {
        'q_item_name_clean': 80,
        'q_description': 200
    }
    file_embed = path_tmp_array / f'embed_{file_raw}_{col}.npy'
    if not file_embed.exists():
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
        embeddings = model.encode(
            df_q[col].to_list(),
            batch_size=8,
            max_length=max_len_dict.get(col, 100),
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']
        np.save(file_embed, embeddings)
    else:
        embeddings = np.load(file_embed)
    print(embeddings.shape)

    df_q = df_q.with_columns(pl.Series(values=embeddings, name='embed'))
    dataset = Dataset.from_polars(df_q)
    dataset.set_format(type='numpy', columns=['embed'], output_all_columns=True)

    path_index = path / 'tmp/index'
    file_index = path_index / f'ip_{file_raw}_{col}.index'
    if not file_index.exists():
        build_index(
            embeddings=embeddings,
            index_path=str(file_index),
            index_infos_path=str(path_index / f'index.json'),
            save_on_disk=True,
            metric_type='ip',
            verbose=30,
        )

    # add index
    dataset.load_faiss_index('embed', path_index / f'ip_{file_raw}_{col}.index')

    score, result = dataset.get_nearest_examples_batch(
        'embed',
        np.asarray(dataset['embed']),
        k=5
    )

    dict_ = {'score': [list(i) for i in score]}
    df_score = pl.DataFrame(dict_)
    df_result = (
        pl.DataFrame(result).drop(['embed'])
        .select(pl.all().name.prefix(f'db_'))
    )

    df_match = pl.concat([df_q.drop(['embed']), df_result, df_score], how='horizontal')
    col_explode = [i for i in df_match.columns if 'db' in i] + ['score']
    df_match = df_match.explode(col_explode)

    df_match.write_parquet(path / f'result_{file_raw}_{col}.parquet')
    print('=> Done \n')


# run('Description_Images_clean', 'q_item_name_clean')
# run('Description_Images_clean', 'q_description')
run('Description_Images_summary', 'q_text_info')
run('Description_Images_summary', 'q_img_info')
run('Description_Images_summary', 'q_combine')
