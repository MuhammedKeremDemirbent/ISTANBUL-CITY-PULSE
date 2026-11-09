
    
    

select
    timestamp as unique_field,
    count(*) as n_records

from "warehouse"."main"."fct_city_pulse_hourly"
where timestamp is not null
group by timestamp
having count(*) > 1


