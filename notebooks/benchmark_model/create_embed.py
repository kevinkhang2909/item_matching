import duckdb
from core_pro.ultilities import make_sync_folder
from pathlib import Path
import sys
sys.path.extend([str(Path.home() / 'PycharmProjects/item_matching')])

from src.item_matching.func.function_image import PipelineImage
from src.item_matching.func.function_text import PipelineText
from src.item_matching.main_match import PipelineMatch, MatchInput
from src.item_matching.pipeline.rerank import ReRank, ReRankConfig


# path
cluster = 'FMCG'
path = make_sync_folder('Item_Matching_Test')

file = path / f'data_sample_{cluster}_0.parquet'
file_inner = path / 'inner.parquet'

# data loading
query = f"""
with base as (
select * exclude(level1_global_be_category, level2_global_be_category, level3_global_be_category)
, level1_global_be_category || '__' || level2_global_be_category || '__' || level3_global_be_category category
from read_parquet('{file}')
)
select *
from base
"""
df = (
    duckdb.sql(query).pl()
    .unique(['item_id'])
)

# text
df = PipelineText().run(df)
df.write_parquet(file_inner)

config = MatchInput(
    ROOT_PATH=path,
    PATH_INNER=file_inner,
    INNER=True ,
    MATCH_BY='text',
    COL_CATEGORY='category',
    TOP_K=10,
    EXPLODE=False,
)
PipelineMatch(config).run()

# image
df, df_img = PipelineImage(path / f'download_img_{cluster}').run(df)
df.write_parquet(file_inner)

config = MatchInput(
    ROOT_PATH=path,
    PATH_INNER=file_inner,
    INNER=True ,
    MATCH_BY='image',
    COL_CATEGORY='category',
    TOP_K=10,
    EXPLODE=False,
)
PipelineMatch(config).run()

# rerank
config = ReRankConfig(ROOTPATH=path, EXPLODE=True)
ReRank(config).run()
