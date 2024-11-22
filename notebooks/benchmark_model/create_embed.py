import duckdb
from core_pro.ultilities import make_sync_folder
from src.item_matching.func.function_image import PipelineImage
from src.item_matching.func.function_text import PipelineText
from src.item_matching.main_match import PipelineMatch, MatchInput


# path
path = make_sync_folder('Item_Matching_Test')
file = path / 'data_sample_ELHA_0.parquet'

query = f"""
with base as (
select * exclude(level1_global_be_category, level2_global_be_category, level3_global_be_category)
, level1_global_be_category || '__' || level2_global_be_category || '__' || level3_global_be_category category
from read_parquet('{file}')
)
select * 
from base
"""
df = duckdb.sql(query).pl()

df = (
    PipelineText().run(df)
    .unique(['item_id'])
)
file_inner = path / 'inner.parquet'
df.write_parquet(file_inner)
    # df, df_img = PipelineImage(path).run(df)

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
