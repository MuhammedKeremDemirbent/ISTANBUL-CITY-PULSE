
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select traffic_index_av
from "warehouse"."main"."fct_city_pulse_hourly"
where traffic_index_av is null



  
  
      
    ) dbt_internal_test