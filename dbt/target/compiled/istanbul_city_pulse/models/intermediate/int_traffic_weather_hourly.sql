with traffic as (
    select * from "warehouse"."main"."stg_traffic"
),

weather as (
    select * from "warehouse"."main"."stg_weather"
)

select
    t.hour_timestamp,
    t.traffic_index_av,
    w.temperature_2m,
    w.precipitation,
    w.wind_speed_10m,
    w.relative_humidity_2m
from traffic t
inner join weather w on t.hour_timestamp = w.hour_timestamp