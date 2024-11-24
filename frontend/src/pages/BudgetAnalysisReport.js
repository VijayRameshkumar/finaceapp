import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import CollapsibleTable from '../components/CollapsibleTable';
import React, { useState, useEffect } from 'react';
import { getUserPermissions, filterBudgetReportData, filterNonbudgetReportData, filterEventReportData } from '../services/api'; // Add specific API calls
import Box from '@mui/material/Box';

const BudgetAnalysisReport = ({ selectedVesselType, selectedVesselSubtype, vesselAgeStart, vesselAgeEnd, categories, subCategories, selectedVessels }) => {
  const [allow, setAllow] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);

  // State for each tab's data and loading status
  const [budgetCatData, setBudgetCatData] = useState([]);
  const [budgetSubcatData, setBudgetSubcatData] = useState([]);
  const [nonBudgetCatData, setNonBudgetCatData] = useState([]);
  const [nonBudgetSubcatData, setNonBudgetSubcatData] = useState([]);
  const [eventSubcatData, setEventSubcatData] = useState([]);


  const [loadingStates, setLoadingStates] = useState({
    budget: false,
    nonBudget: false,
    event: false
  });

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = localStorage.getItem('token');
        const user_id = localStorage.getItem('id');
        if (token) {
          const response = await getUserPermissions(user_id);
          setAllow(response.data.can_download);
        }
      } catch (error) {
        console.error("Error fetching user data:", error);
      }
    };
    fetchUserData();
  }, []);


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
        if (selectedIndex === 0) {
          setLoadingStates(prev => ({ ...prev, budget: true }));
          const response = await filterBudgetReportData(filterParams);  // Fetch for budget categories
          setBudgetCatData(response.data.budget_cat_data);
          setBudgetSubcatData(response.data.budget_subcat_data);
          setLoadingStates(prev => ({ ...prev, budget: false }));
        } else if (selectedIndex === 1) {
          setLoadingStates(prev => ({ ...prev, nonBudget: true }));
          const response = await filterNonbudgetReportData(filterParams); // Fetch for non-budget categories
          setNonBudgetCatData(response.data.non_budget_cat_data);
          setNonBudgetSubcatData(response.data.nonbudget_subcat_data);
          setLoadingStates(prev => ({ ...prev, nonBudget: false }));
        } else if (selectedIndex === 2) {
          setLoadingStates(prev => ({ ...prev, event: true }));
          const response = await filterEventReportData(filterParams);  // Fetch for event categories
          setEventSubcatData(response.data.event_subcat_data);
          setLoadingStates(prev => ({ ...prev, event: false }));
        }
      } catch (error) {
        console.error("Failed to load data for tab:", error); // Log error to console
        alert("Please try again."); // Notify the user
        // Reset loading states in case of error
        setLoadingStates(prev => ({ ...prev, budget: false, nonBudget: false, event: false }));
      }
    };

    loadDataForTab();
  }, [selectedIndex, selectedVesselType, selectedVesselSubtype, vesselAgeStart, vesselAgeEnd, categories, subCategories, selectedVessels]);


  return (
    <div>
      <hr className="my-6 border-gray-300" />
      {/* <h2 className="text-2xl font-bold pb-5">Budget Analysis - Report View</h2> */}
      <Tabs selectedIndex={selectedIndex} onSelect={index => setSelectedIndex(index)}>
        <TabList style={{ backgroundColor: 'white', boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.2)' }}>
          <Tab style={{ backgroundColor: selectedIndex === 0 ? '#041085' : 'white', color: selectedIndex === 0 ? 'white' : 'black', boxShadow: selectedIndex === 0 ? '0px 4px 10px rgba(0, 0, 0, 0.2)' : 'none' }}>
            Budget Categories
          </Tab>
          <Tab style={{ backgroundColor: selectedIndex === 1 ? '#041085' : 'white', color: selectedIndex === 1 ? 'white' : 'black', boxShadow: selectedIndex === 0 ? '0px 4px 10px rgba(0, 0, 0, 0.2)' : 'none' }}>
            Additional Categories
          </Tab>
          <Tab style={{ backgroundColor: selectedIndex === 2 ? '#041085' : 'white', color: selectedIndex === 2 ? 'white' : 'black', boxShadow: selectedIndex === 0 ? '0px 4px 10px rgba(0, 0, 0, 0.2)' : 'none' }}>
            Event Categories
          </Tab>
        </TabList>

        {/* Tab for the Excel Template */}
        <TabPanel>
          <Box sx={{
            backgroundColor: 'white',
            borderRadius: 2,
            boxShadow: 2,
            padding: 2,
          }}>
            <h2 className="text-1xl font-bold pb-5 pt-2">Category level per month</h2>
            <div>
              {(budgetCatData.length === 0) ? (
                <p className="text-red-500 pl-5">Loading...</p>
              ) : (
                <CollapsibleTable data={budgetCatData} allow={allow} />
              )}
            </div>
          </Box>
          {/* Horizontal Divider */}
          <hr className="my-6 border-gray-100" />
          <Box sx={{
            backgroundColor: 'white',
            borderRadius: 2,
            boxShadow: 2,
            padding: 2,
          }}>

            <h2 className="text-1xl font-bold pb-5 pt-5">Sub-Category level per month</h2>
            <div>
              {(budgetSubcatData.length === 0) ? (
                <p className="text-red-500 pl-5">Loading...</p>
              ) : (
                <CollapsibleTable data={budgetSubcatData} allow={allow} />
              )}
            </div>
          </Box>
        </TabPanel>

        <TabPanel>
          <Box sx={{
            backgroundColor: 'white',
            borderRadius: 2,
            boxShadow: 2,
            padding: 2,
          }}>
            <h2 className="text-1xl font-bold pb-5 pt-5">Category level per month</h2>
            <div>
              {(nonBudgetCatData.length === 0) ? (
                <p className="text-red-500 pl-5">Loading...</p>
              ) : (
                <CollapsibleTable data={nonBudgetCatData} allow={allow} />
              )}
            </div>
          </Box>
          <hr className="my-6 border-gray-100" />
          <Box sx={{
            backgroundColor: 'white',
            borderRadius: 2,
            boxShadow: 2,
            padding: 2,
          }}>

            <h2 className="text-1xl font-bold pb-5 pt-5">Sub-Category level per month</h2>
            <div>
              {(nonBudgetSubcatData.length === 0) ? (
                <p className="text-red-500 pl-5">Loading...</p>
              ) : (
                <CollapsibleTable data={nonBudgetSubcatData} allow={allow} />
              )}
            </div>
          </Box>
        </TabPanel>

        {/* Tab for the PDF Report */}
        <TabPanel>
          <Box sx={{
            backgroundColor: 'white',
            borderRadius: 2,
            boxShadow: 2,
            padding: 2,
          }}>
            <h2 className="text-1xl font-bold pb-5 pt-5">Sub-Category level per month</h2>
            <div>
              {(eventSubcatData.length === 0) ? (
                <p className="text-red-500 pl-5">Loading...</p>
              ) : (
                <CollapsibleTable data={eventSubcatData} allow={allow} />
              )}
            </div>
          </Box>
        </TabPanel>
      </Tabs>
    </div>
  );
};

export default BudgetAnalysisReport;
