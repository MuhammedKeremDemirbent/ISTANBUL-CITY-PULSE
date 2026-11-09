with source as (
    select * from {{ source('silver', 'traffic') }}
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