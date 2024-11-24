import React, { useState, useEffect } from 'react';
import { getVesselTypes, getVesselSubtypes, filterInpuyts, filterReportData } from '../services/api';
import BudgetAnalysisTrend from './BudgetAnalysisTrend';
import BudgetAnalysisReport from './BudgetAnalysisReport';
import Slider from '@mui/material/Slider';
import { useTheme } from '@mui/material/styles';
import OutlinedInput from '@mui/material/OutlinedInput';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import BarChart from '../components/BarCharts';
import Typography from '@mui/material/Typography';
import Tooltip from '@mui/material/Tooltip';
import Box from '@mui/material/Box'; // Import Box from Material UI
import ListItemText from '@mui/material/ListItemText';
import Checkbox from '@mui/material/Checkbox';
import { ExpandMore, ExpandLess } from '@mui/icons-material';

const BudgetAnalysis = () => {
  const theme = useTheme();
  const [currentView, setCurrentView] = useState('report');

  const [vesselTypes, setVesselTypes] = useState(['BULK CARRIER']);
  const [vesselSubtypes, setVesselSubtypes] = useState(['HANDYSIZE']);
  const [categories, setCategories] = useState(['Select All']);
  const [subCategories, setSubCategories] = useState(['Select All']);
  const [vessels, setVessels] = useState(['Select All']);
  const [vesselAgeStart, setVesselAgeStart] = useState(3);
  const [vesselAgeEnd, setVesselAgeEnd] = useState(6);
  const [ageValue, setAgeValue] = useState([3,6]);
  const [vesselsCount, setVesselsCount] = useState(0);
  const [vesselAgeData, setVesselAgeData] = useState({});
  const [selectedVesselType, setSelectedVesselType] = useState('BULK CARRIER');
  const [selectedVesselSubtype, setSelectedVesselSubtype] = useState(['HANDYSIZE']);
  const [selectedCategories, setSelectedCategories] = useState(['Select All']);
  const [selectedSubCategories, setSelectedSubCategories] = useState(['Select All']);
  const [selectedVessels, setSelectedVessels] = useState(['Select All']);


  const [inputsCollapsed, setInputsCollapsed] = useState(false);

  function valuetext(value) {
    return `${value}`;
  }

   // Predefined ranges
   const ranges = [
    [1, 3], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10],
    [8, 11], [9, 12], [10, 13], [11, 14], [12, 15], [13, 16],
    [14, 17], [15, 18], [16, 19], [17, 20]
  ];

  // Function to find the corresponding end for a given start or vice versa
  const findMatchingRange = (start, end) => {
    // If start is given, find the corresponding end
    if (start !== null) {
      const range = ranges.find(([s]) => s === start);
      return range ? range[1] : end;
    }
    // If end is given, find the corresponding start
    if (end !== null) {
      const range = ranges.find(([, e]) => e === end);
      return range ? range[0] : start;
    }
    return [start, end];
  };

  // Handle changes to either start or end of the range
  const handleAgeChange = (event, newValue) => {
    let [newStart, newEnd] = newValue;

    // Ensure start doesn't go beyond 17 and end isn't below 3
    if (newStart >= 18) newStart = 17;
    if (newEnd <= 2) newEnd = 3;

    // Update the other value based on the predefined ranges
    if (newStart !== ageValue[0]) {
      newEnd = findMatchingRange(newStart, null); // Adjust end based on new start
    } else if (newEnd !== ageValue[1]) {
      newStart = findMatchingRange(null, newEnd); // Adjust start based on new end
    }

    // Update the state with the validated range
    setAgeValue([newStart, newEnd]);
    setVesselAgeStart(newStart);
    setVesselAgeEnd(newEnd);
  };

 

  const marks = [
    { value: 1, label: '1' },
    { value: 20, label: '20' },
  ];

  const MenuProps = {
    PaperProps: {
      style: {
        maxHeight: 100,
        width: 250,
      },
    },
  };

  function getStyles(name, values, theme) {
    return {
      fontWeight: values.includes(name)
        ? theme.typography.fontWeightMedium
        : theme.typography.fontWeightRegular,
    };
  }

  const categoriesHandleChange = (event) => {
    const { target: { value } } = event;
    if (value[value.length - 1] === 'Select All') {
      setSelectedCategories(['Select All']);
    } else {
      setSelectedCategories(value.filter((item) => item !== 'Select All'));
    }
  };
  

  const subTypesHandleChange = (event) => {
    const { target: { value } } = event;
    setSelectedVesselSubtype(typeof value === 'string' ? value.split(',') : value);
    // fetchFilterReportData();
  };

  const subCategoriesHandleChange = (event) => {
    const { target: { value } } = event;
    // fetchFilterReportData();
    if (value[value.length - 1] === 'Select All') {
      setSelectedSubCategories(['Select All']);
    } else {
      setSelectedSubCategories(value.filter((item) => item !== 'Select All'));
    }

  };

  const selectedVesselsHandleChange = (event) => {
    const { target: { value } } = event;
    if (value[value.length - 1] === 'Select All') {
      setSelectedVessels(['Select All']);
    } else {
      setSelectedVessels(value.filter((item) => item !== 'Select All'));
    }
  };

  useEffect(() => {
    fetchVesselTypesData();
    fetchVesselSubtypesData(selectedVesselType);
    // fetchFilterReportData();
    fetchInputsData();
  }, []);

  const fetchVesselTypesData = async () => {
    try {
      const response = await getVesselTypes();
      setVesselTypes(response.data);
    } catch (error) {
      console.error('Error fetching vessel types:', error);
    }
  };

  const fetchVesselSubtypesData = async (vesselType) => {
    try {
      const response = await getVesselSubtypes(vesselType);
      setVesselSubtypes(response.data);
    } catch (error) {
      console.error('Error fetching vessel subtypes:', error);
    }
  };

  const fetchInputsData = async () => {
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
      const response = await filterInpuyts(filterParams);

      setCategories(response.data.vessel_cat_options);
      setSubCategories(response.data.vessel_subcat_options);
      setVessels(response.data.selected_vessels_option);
      setVesselsCount(response.data.vessels_selected_count);
      setVesselAgeData(response.data.age_count_data);
    } catch (error) {
      console.error('Error fetching report data:', error);
    }
  };



  const handleVesselTypeChange = (event) => {
    const _vesselType = event.target.value;
    setSelectedVesselType(_vesselType);
    fetchVesselSubtypesData(_vesselType);
  };

  const handleViewSwitch = (view) => {
    setCurrentView(view);
  };

  const toggleInputs = () => {
    setInputsCollapsed(!inputsCollapsed);
  };

  return (
    <div className="p-4 relative" style={{ backgroundColor: '#f0f4f8' }}>
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-blue-900">Budget Analysis</h2>
        <div className="flex items-center space-x-4">
          <button
            onClick={() => handleViewSwitch('report')}
            className={`px-4 py-2 rounded-md ${currentView === 'report' ? 'bg-blue-900 text-white' : 'bg-gray-200'}`}
          >
            Report
          </button>
          <div className="h-6 w-px bg-gray-400"></div>
          <button
            onClick={() => handleViewSwitch('trend')}
            className={`px-4 py-2 rounded-md ${currentView === 'trend' ? 'bg-blue-900 text-white' : 'bg-gray-200'}`}
          >
            Trend
          </button>
        </div>
      </div>
      <p className="mt-4">Synergy Budget Analysis Tool Version 1.0</p>
      <hr className="my-6 border-gray-300" />

      <div onClick={toggleInputs} className="cursor-pointer flex items-center">
        <h3 className="text-xl font-bold">Filters</h3>
        {inputsCollapsed ? <ExpandMore /> : <ExpandLess />}
      </div>

      {!inputsCollapsed && (
        <Box sx={{
          backgroundColor: 'white',
          borderRadius: 2,
          boxShadow: 2,
          padding: 2,
        }}>
          <div className="grid grid-cols-3 gap-4 mt-4">
            <FormControl fullWidth size="small" sx={{ maxWidth: 250, marginBottom: 2 }}>
              <InputLabel>Vessel Type</InputLabel>
              <Select
                value={selectedVesselType}
                onChange={handleVesselTypeChange}
                input={<OutlinedInput label="Vessel Type" />}
                MenuProps={MenuProps}
              >
                {vesselTypes.map((type) => (
                  <MenuItem key={type} value={type}>
                    {type}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth size="small" sx={{ maxWidth: 250, marginBottom: 2 }}>
              <InputLabel>Vessel Subtype</InputLabel>
              <Tooltip title={selectedVesselSubtype.length > 0 ? selectedVesselSubtype.join(', ') : 'Select Sub Vessel Type'}>
              <Select
                multiple
                value={selectedVesselSubtype}
                onChange={subTypesHandleChange}
                input={<OutlinedInput label="Vessel Subtype" />}
                renderValue={(selected) => selected.join(', ')}
                MenuProps={MenuProps}
              >
                {vesselSubtypes.map((subtype) => (
                  <MenuItem
                    key={subtype}
                    value={subtype}
                    style={getStyles(subtype, selectedVesselSubtype, theme)}
                  >
                    <Checkbox checked={selectedVesselSubtype.includes(subtype)} />
                    <ListItemText primary={subtype} />
                  </MenuItem>
                ))}
              </Select>
              </Tooltip>
            </FormControl>

            <FormControl fullWidth size="small" sx={{ maxWidth: 250, marginBottom: 2 }}>
              <InputLabel id="categories-checkbox-label">Categories</InputLabel>
              <Tooltip title={selectedCategories.length > 0 ? selectedCategories.join(', ') : 'Select Categories'}>
              <Select
                labelId="categories-checkbox-label"
                id="categories-checkbox"
                multiple
                value={selectedCategories}
                onChange={categoriesHandleChange}
                input={<OutlinedInput label="Categories" />}
                renderValue={(selected) => selected.join(', ')}
                MenuProps={MenuProps}
              >
                {categories.map((category) => (
                  <MenuItem key={category} value={category}>
                    <Checkbox checked={selectedCategories.includes(category)} />
                    <ListItemText primary={category} />
                  </MenuItem>
                ))}
              </Select>
              </Tooltip>
            </FormControl>

            <FormControl fullWidth size="small" sx={{ maxWidth: 250, marginBottom: 2 }}>
              <InputLabel>Sub-Categories</InputLabel>
              <Tooltip title={selectedSubCategories.length > 0 ? selectedSubCategories.join(', ') : 'Select Sub Category'}>
              <Select
                multiple
                value={selectedSubCategories}
                onChange={subCategoriesHandleChange}
                input={<OutlinedInput label="Sub-Categories" />}
                MenuProps={MenuProps}
                renderValue={(selected) => selected.join(', ')}
              >
                {subCategories.map((subCategory) => (
                  <MenuItem
                    key={subCategory}
                    value={subCategory}
                    style={getStyles(subCategory, selectedSubCategories, theme)}
                  >
                   <Checkbox checked={selectedSubCategories.includes(subCategory)} />
                   <ListItemText primary={subCategory} />
                  </MenuItem>
                ))}
              </Select>
              </Tooltip>
            </FormControl>

            <FormControl fullWidth size="small" sx={{ maxWidth: 250, marginBottom: 2 }}>
              <InputLabel>Vessels</InputLabel>
              <Tooltip title={selectedVessels.length > 0 ? selectedVessels.join(', ') : 'Select Vessel Type'}>
              <Select
                multiple
                value={selectedVessels}
                onChange={selectedVesselsHandleChange}
                input={<OutlinedInput label="Vessels" />}
                MenuProps={MenuProps}
                renderValue={(selected) => selected.join(', ')}
              >
                {vessels.map((vessel) => (
                  <MenuItem
                    key={vessel}
                    value={vessel}
                    style={getStyles(vessel, selectedVessels, theme)}
                  >
                     <Checkbox checked={selectedVessels.includes(vessel)} />
                     <ListItemText primary={vessel} />
                  </MenuItem>
                ))}
              </Select>
              </Tooltip>
            </FormControl>

            <div>
              <Typography
                id="non-linear-slider"
                gutterBottom
                sx={{ color: 'gray', fontSize: '.8rem', marginBottom: '-5px' }}
              >
                Vessel Age
              </Typography>
              <Box sx={{ width: '250px', marginLeft: '5px' }}> {/* Adjust width as needed */}
                <Slider
                  getAriaLabel={() => 'Age Range'}
                  value={ageValue}
                  onChange={handleAgeChange}
                  valueLabelDisplay="auto"
                  getAriaValueText={valuetext}
                  marks={marks}
                  min={1}
                  max={20}
                />
              </Box>

            </div>
          </div>
        </Box>
      )}

      <hr className="my-6 border-gray-300" />
      <Box sx={{
        backgroundColor: 'white',
        borderRadius: 2,
        boxShadow: 2,
        padding: 2,
      }}>
        <h2 className="text-xl font-bold pb-5">Analysis</h2>
        <BarChart vesselAgeData={vesselAgeData} />
      </Box>
      {/* Horizontal Divider */}
      <hr className="my-6 border-gray-100" />

      <div className="mt-3">
        {currentView === 'report' && (
          <BudgetAnalysisReport
            selectedVesselType={selectedVesselType}
            selectedVesselSubtype={selectedVesselSubtype}
            vesselAgeStart={vesselAgeStart}
            vesselAgeEnd={vesselAgeEnd}
            categories={categories}
            subCategories={subCategories}
            selectedVessels={selectedVessels}
          />
        )}
        {currentView === 'trend' && (
          <BudgetAnalysisTrend
            selectedVesselType={selectedVesselType}
            selectedVesselSubtype={selectedVesselSubtype}
            vesselAgeStart={vesselAgeStart}
            vesselAgeEnd={vesselAgeEnd}
            categories={categories}
            subCategories={subCategories}
            selectedVessels={selectedVessels}
          />
        )}
      </div>

    </div>
  );
};

export default BudgetAnalysis;
