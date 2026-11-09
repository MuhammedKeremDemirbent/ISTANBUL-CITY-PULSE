with source as (
    select * from {{ source('silver', 'weather') }}
)

select
    date_trunc('hour', timestamp) as hour_timestamp,
    cast(temperature_2m as double) as temperature_2m,
    cast(precipitation as double) as precipitation,
    cast(wind_speed_10m as double) as wind_speed_10m,
    cast(relative_humidity_2m as double) as relative_humidity_2m
from source