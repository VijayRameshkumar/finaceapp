import QuartilesLineChart from '../components/QuartilesLineChart';
import Box from '@mui/material/Box';
import { filterTrendData } from '../services/api'; // Add specific API calls
import React, { useState, useEffect } from 'react';

const BudgetAnalysisTrend = ({selectedVesselType,selectedVesselSubtype,vesselAgeStart,vesselAgeEnd,categories,subCategories,selectedVessels}) => {
 
    const [plotlyMonthlyQuartilesData, setPlotlyMonthlyQuartilesData] = useState([]);
    const [plotlyYearlyQuartilesData, setPlotlyYearlyQuartilesData] = useState([]);

  // Load data only when a tab is selected
  useEffect(() => {

    const filterParams = {
      vessel_type: selectedVesselType,
      vessel_subtype: selectedVesselSubtype,
      vessel_age_start: vesselAgeStart,
      vessel_age_end: vesselAgeEnd,
      vessel_cat: categories,
      vessel_subcat: subCategories,
      selected_vessels_dropdown: selectedVessels,
    };

    const loadDataForTab = async () => {
   
            try {
              const filterParams = {
                vessel_type: selectedVesselType,
                vessel_subtype: selectedVesselSubtype,
                vessel_age_start: vesselAgeStart,
                vessel_age_end: vesselAgeEnd,
                vessel_cat: categories,
                vessel_subcat: subCategories,
                selected_vessels_dropdown: selectedVessels,
              };
        
              const response = await filterTrendData(filterParams);
        
              setPlotlyMonthlyQuartilesData(response.data.plotly_monthly_quartiles_data);
              setPlotlyYearlyQuartilesData(response.data.plotly_yearly_quartiles_data);
            } catch (error) {
              console.error('Error fetching report data:', error);
            }
          
    };
    loadDataForTab();
  }, [selectedVesselType,selectedVesselSubtype,vesselAgeStart,vesselAgeEnd,categories,subCategories,selectedVessels]);

    return (
        <div>
            {/* <h2 className="text-2xl font-bold">Budget Analysis - Trend View</h2> */}
            <h2 className="text-1xl font-bold pb-5 pt-10">1. Monthly Trend - PER MONTH</h2>
            <div>
                {(plotlyMonthlyQuartilesData == []) ? (
                    <p className="text-red-500 pl-5">Data is Loading...</p>
                ) : (
                    <QuartilesLineChart data={plotlyMonthlyQuartilesData} />
                )}
            </div>
            <h2 className="text-1xl font-bold pb-5 pt-10">2. Yearly - PER MONTH</h2>
            <div>
                {(plotlyYearlyQuartilesData == []) ? (
                    <p className="text-red-500 pl-5">Data is Loading...</p>
                ) : (
                    <QuartilesLineChart data={plotlyYearlyQuartilesData} />
                )}
            </div>
        </div>
    );
};

export default BudgetAnalysisTrend;
