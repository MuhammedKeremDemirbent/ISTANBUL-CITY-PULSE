
  
  create view "warehouse"."main"."stg_traffic__dbt_tmp" as (
    with source as (
    select * from 'data/silver/traffic_hourly/istanbul_2023-01-01_2025-12-31.parquet'
),

cleaned as (
    select
        date_trunc('hour', timestamp) as hour_timestamp,
        cast(traffic_index_av as double) as traffic_index_av,
        row_number() over (
            partition by date_trunc('hour', timestamp) 
            order by timestamp desc
        ) as rn
    from source
    where traffic_index_av is not null
)

select
    hour_timestamp,
    traffic_index_av
from cleaned
where rn = 1
  );
