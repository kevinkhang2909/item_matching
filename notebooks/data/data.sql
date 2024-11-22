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
,base as (
    select distinct
        o.item_id
        ,o.item_name
        ,o.shop_id
        ,o.shop_name
        ,o.level1_global_be_category
        ,o.level2_global_be_category
        ,o.level3_global_be_category
        ,m.cluster
    from
        mp_order.dwd_order_item_all_ent_df__vn_s0_live o
        left join cat m on m.level1_global = o.level1_global_be_category
    where
        o.grass_date = current_date - interval '1' day
        and m.cluster = '{0}'
    limit
        100000
)
select distinct
    o.*
    ,p.description
    ,p.images
from
    base o
    left join mp_item.dim_item_ext__vn_s0_live p on o.item_id = p.item_id
    and p.grass_date = current_date - interval '1' day