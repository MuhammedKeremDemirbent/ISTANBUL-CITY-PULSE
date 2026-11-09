with source as (
    select * from 'data/silver/holidays/turkey_holidays_2023_2025.parquet'
)

select
    cast(date as date) as holiday_date
from source