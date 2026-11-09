with source as (
    select * from {{ source('silver', 'holidays') }}
)

select
    cast(date as date) as holiday_date
from source
