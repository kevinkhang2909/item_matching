from core_pro import DataPipeLine
from core_pro.ultilities import make_sync_folder


query = f"""
with base as (
    select distinct
        level1_global_be_category
        ,final_brand
    from mp_mall.dim_brand_item_brandmapping__vn_s0_live
    where
        grass_date = current_date - interval '1' day
        and global_brand != 'NoBrand'
        and global_brand is not null
        and status = 1
)
select
    *
from base
"""
path = make_sync_folder('brand')
df = DataPipeLine(query).run_presto_to_df(save_path=path / 'brand.parquet')
