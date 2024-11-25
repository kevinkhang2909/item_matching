from core_pro import DataPipeLine
from core_pro.ultilities import make_sync_folder, update_df
import polars as pl


# download parquet
path = make_sync_folder('Item_Matching_Test')
sh = '1JQiN33Z8hTPd_ENBAoqToMKXGr2Qp5ZU0rjvFPHCTAI'

query = f"""
with cat as (
    select distinct
        level1_global
        ,cluster
    from vnbi_mkt.general__global_category_map
    where
        ingestion_timestamp in (
            select
                max(ingestion_timestamp)
            from vnbi_mkt.general__global_category_map
        )
)
select cluster
    ,o.level1_global_be_category
    ,o.level2_global_be_category
    ,o.level3_global_be_category
    ,count(distinct case when o.grass_date >= current_date - interval '30' day then o.item_id else null end) total_items_30d
    ,count(distinct case when o.grass_date >= current_date - interval '60' day then o.item_id else null end) total_items_60d
    ,count(distinct o.item_id) total_items_90d
from
    mp_order.dwd_order_item_all_ent_df__vn_s0_live o
    left join cat m on m.level1_global = o.level1_global_be_category
where
    o.grass_date >= current_date - interval '90' day
group by 1, 2, 3, 4
"""
save_path = path / 'stats/stats_l3.parquet'
# df = DataPipeLine(query).run_presto_to_df(save_path=save_path)

df = (
    pl.read_parquet(save_path)
    .with_columns(
        (pl.col('total_items_90d') / pl.col('total_items_90d').sum()).alias('pct_90d'),
        (pl.col('total_items_60d') / pl.col('total_items_30d')).alias('dif_60'),
        (pl.col('total_items_90d') / pl.col('total_items_30d')).alias('dif_90'),
    )
    .drop_nulls(subset=['cluster'])
    .sort(['level1_global_be_category'])
)
update_df(df, 'category', sh, start='A1')

query = f"""
with cat as (
    select distinct
        level1_global
        ,cluster
    from vnbi_mkt.general__global_category_map
    where
        ingestion_timestamp in (
            select
                max(ingestion_timestamp)
            from vnbi_mkt.general__global_category_map
        )
)
select cluster
    ,o.level1_global_be_category
    ,o.level2_global_be_category
    ,o.level3_global_be_category
    ,o.level4_global_be_category
    ,o.level5_global_be_category
    ,count(distinct case when o.grass_date >= current_date - interval '30' day then o.item_id else null end) total_items_30d
    ,count(distinct case when o.grass_date >= current_date - interval '60' day then o.item_id else null end) total_items_60d
    ,count(distinct o.item_id) total_items_90d
from
    mp_order.dwd_order_item_all_ent_df__vn_s0_live o
    left join cat m on m.level1_global = o.level1_global_be_category
where
    o.grass_date >= current_date - interval '90' day
group by 1, 2, 3, 4, 5, 6
"""
save_path = path / 'stats/stats_l5.parquet'
# df = DataPipeLine(query).run_presto_to_df(save_path=save_path)

df = (
    pl.read_parquet(save_path)
    .with_columns(
        (pl.col('total_items_90d') / pl.col('total_items_90d').sum()).alias('pct_90d'),
        (pl.col('total_items_60d') / pl.col('total_items_30d')).alias('dif_60'),
        (pl.col('total_items_90d') / pl.col('total_items_30d')).alias('dif_90'),
    )
    .drop_nulls(subset=['cluster'])
    .sort(['level1_global_be_category'])
)
update_df(df, 'category', sh, start='M1')
