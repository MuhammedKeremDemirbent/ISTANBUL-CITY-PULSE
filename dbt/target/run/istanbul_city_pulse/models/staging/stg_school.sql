
  
  create view "warehouse"."main"."stg_school__dbt_tmp" as (
    with source as (
    select * from 'data/silver/school/school_calendar_2023_2025.parquet'
)

select
    date_trunc('hour', timestamp) as hour_timestamp,
    cast(is_school_term as integer) as is_school_term,
    cast(is_school_morning_rush as integer) as is_school_morning_rush,
    cast(is_school_afternoon_rush as integer) as is_school_afternoon_rush
from source
  );
