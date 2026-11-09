
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

select
    timestamp as unique_field,
    count(*) as n_records

from "warehouse"."main"."fct_city_pulse_hourly"
where timestamp is not null
group by timestamp
having count(*) > 1



  
  
      
    ) dbt_internal_test