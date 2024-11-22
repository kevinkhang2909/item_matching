from pathlib import Path
import duckdb


path = Path('/Users/kevinkhang/Downloads/Data')

file_brand = path / 'brand/brand.parquet'
file_data = path / 'Item_Matching_Test/clean.parquet'

query = f"""
select level1_global_be_category
 , array_agg(distinct lower(final_brand)) final_brand
 from read_parquet('{file_brand}')
 group by 1
"""
df_l1_brand = duckdb.sql(query).pl()

query = f"""
select final_brand
 , array_agg(distinct lower(level1_global_be_category)) level1_global_be_category
 , len(array_agg(distinct lower(level1_global_be_category))) len
 from read_parquet('{file_brand}')
 group by 1
"""
df_brand = duckdb.sql(query).pl()

# query = f"""
# with brand as (
#     select level1_global_be_category
#      , lower(final_brand) final_brand
#      from read_parquet('{file_brand}')
# )
# select i.*
# , b.final_brand
# from read_parquet('{file_data}') i
# left join brand as b
# on i.level1_global_be_category = b.level1_global_be_category
# """
# df = duckdb.sql(query).pl()
# print(df.shape)
