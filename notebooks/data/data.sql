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
        and cluster = '{0}'
)
select distinct
    o.item_id
    ,o.item_name
    ,o.shop_id
    ,o.shop_name
    ,o.level1_global_be_category
    --,o.level2_global_be_category
    --,o.level3_global_be_category
    ,m.cluster
    --,p.description
    ,p.images
from
    mp_order.dwd_order_item_all_ent_df__vn_s0_live o
    join cat m on m.level1_global = o.level1_global_be_category
    left join mp_item.dim_item_ext__vn_s0_live p on o.item_id = p.item_id
    and p.grass_date = current_date - interval '2' day
where
    o.grass_date = current_date - interval '1' day
    and o.is_bi_excluded = 0
    and o.is_net_order = 1
limit
    10000