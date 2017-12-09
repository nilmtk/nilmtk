nilmtk\.metergroup
==================

.. topic:: Module summary

    .. automodule:: nilmtk.metergroup
       :no-members:
       :no-undoc-members:
       :no-show-inheritance: 

       .. rubric:: Functions

       .. autosummary::

          combine_chunks_from_generators
          iterate_through_submeters_of_two_metergroups
          meter_sorting_key
          replace_dataset

    
       .. rubric:: Classes

       .. autosummary::
        
          MeterGroup
          MeterGroupID
    


    


    
        

.. topic:: Class: MeterGroup

    .. autoclass:: MeterGroup
       :no-members:
       :no-undoc-members:
            
       .. Note:: This class has inherited methods, see the base class(es) for additional methods
            
            
       .. rubric:: Methods

       .. autosummary::
                
            all_meters
            available_ac_types
            available_physical_quantities
            building
            call_method_on_all_meters
            clear_cache
            contains_meters_from_multiple_buildings
            correlation_of_sum_of_submeters_with_mains
            dataframe_of_meters
            dataset
            describe
            dominant_appliance
            dominant_appliances
            draw_wiring_graph
            dropout_rate
            energy_per_meter
            entropy_per_meter
            fraction_per_meter
            from_list
            get_labels
            get_timeframe
            good_sections
            groupby
            import_metadata
            instance
            is_site_meter
            label
            load
            mains
            matches
            meters_directly_downstream_of_mains
            nested_metergroups
            pairwise
            pairwise_correlation
            pairwise_mutual_information
            plot
            plot_good_sections
            plot_multiple
            plot_when_on
            proportion_of_energy_submetered
            proportion_of_upstream_total_per_meter
            sample_period
            select
            select_top_k
            select_using_appliances
            simultaneous_switches
            sort_meters
            submeters
            total_energy
            train_test_split
            union
            upstream_meter
            use_alternative_mains
            values_for_appliance_metadata_key
            wiring_graph
            
            
       .. rubric:: Attributes

       .. autosummary::
                
            appliances
            identifier
            

.. topic:: Class: MeterGroupID

    .. autoclass:: MeterGroupID
       :no-members:
       :no-undoc-members:
            
            
            
       .. rubric:: Attributes

       .. autosummary::
                
            meters
            
    


.. rubric:: Module detail

.. automodule:: nilmtk.metergroup
    :members:
    :undoc-members:
    :show-inheritance:


