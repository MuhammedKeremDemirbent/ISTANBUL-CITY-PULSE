
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select timestamp
from "warehouse"."main"."fct_city_pulse_hourly"
where timestamp is null



  
  
      
    ) dbt_internal_test