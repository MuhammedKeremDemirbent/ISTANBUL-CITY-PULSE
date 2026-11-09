with base as (
    select * from "warehouse"."main"."int_traffic_weather_hourly"
),

holidays as (
    select * from "warehouse"."main"."stg_holidays"
),

school as (
    select * from "warehouse"."main"."stg_school"
)

select
    b.hour_timestamp as timestamp,
    date_part('hour', b.hour_timestamp) as hour,
    -- Pazartesi: 1, Pazar: 7 (ISO standardı)
    date_part('isodow', b.hour_timestamp) as day_of_week,
    date_part('month', b.hour_timestamp) as month,
    case when date_part('isodow', b.hour_timestamp) in (6, 7) then 1 else 0 end as is_weekend,
    case when h.holiday_date is not null then 1 else 0 end as is_holiday,
    
    b.temperature_2m,
    b.precipitation,
    case when b.precipitation > 0.0 then 1 else 0 end as is_raining,
    case when b.precipitation >= 5.0 then 1 else 0 end as heavy_rain,
    b.wind_speed_10m,
    b.relative_humidity_2m,
    
    -- Zaman anahtarlı Lag öznitelikleri (1 saat ve 24 saat öncesi)
    lag(b.traffic_index_av, 1) over (order by b.hour_timestamp) as traffic_index_av_lag_1h,
    lag(b.traffic_index_av, 24) over (order by b.hour_timestamp) as traffic_index_av_lag_24h,
    
    coalesce(s.is_school_term, 0) as is_school_term,
    coalesce(s.is_school_morning_rush, 0) as is_school_morning_rush,
    coalesce(s.is_school_afternoon_rush, 0) as is_school_afternoon_rush,
    
    b.traffic_index_av

from base b
left join holidays h on cast(b.hour_timestamp as date) = h.holiday_date
left join school s on b.hour_timestamp = s.hour_timestamp