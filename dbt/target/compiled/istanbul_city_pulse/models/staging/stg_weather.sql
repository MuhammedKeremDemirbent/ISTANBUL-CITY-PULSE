with source as (
    select * from 'data/silver/weather_hourly/istanbul_center_2023-01-01_2025-12-31.parquet'
)

select
    date_trunc('hour', timestamp) as hour_timestamp,
    cast(temperature_2m as double) as temperature_2m,
    cast(precipitation as double) as precipitation,
    cast(wind_speed_10m as double) as wind_speed_10m,
    cast(relative_humidity_2m as double) as relative_humidity_2m
from source