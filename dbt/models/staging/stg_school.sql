with source as (
    select * from {{ source('silver', 'school') }}
)

select
    date_trunc('hour', timestamp) as hour_timestamp,
    cast(is_school_term as integer) as is_school_term,
    cast(is_school_morning_rush as integer) as is_school_morning_rush,
    cast(is_school_afternoon_rush as integer) as is_school_afternoon_rush
from source