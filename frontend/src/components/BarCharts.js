import * as React from 'react';
import { BarChart } from '@mui/x-charts/BarChart';

export default function BarCharts({ vesselAgeData }) {
  // Ensure that vesselAgeData exists and contains valid x and y arrays
  if (!vesselAgeData || !Array.isArray(vesselAgeData.x) || !Array.isArray(vesselAgeData.y)) {
    return <div><p className="text-red-500 pl-5">Loading...</p></div>; // Show a message when data is missing or invalid
  }

  // Destructure age (x) and count (y) arrays from vesselAgeData
  const { x: ages, y: counts } = vesselAgeData;

  // Filter ages and counts where age is between 0 and 20
  const formattedData = ages.reduce((acc, age, index) => {
    if (age >= 0 && age <= 20) {
      acc.push({ age, count: counts[index] });
    }
    return acc;
  }, []);

  // Generate an array of x-axis ticks: [0, 2, 4, ..., 20]
  const xTicks = Array.from({ length: 11 }, (_, i) => i * 2); 

  return (
    <BarChart
      dataset={formattedData} // Provide the transformed dataset
      series={[
        { dataKey: 'count', label: 'Vessel Count' } // Define the dataKey for counts
      ]}
      xAxis={[{
        scaleType: 'band', 
        dataKey: 'age', 
        label: 'Vessel Age', 
        ticks: xTicks // Set custom tick values for x-axis
      }]} 
      yAxis={[{ label: 'Vessel Count' }]} // Label for y-axis
      width={800}
      height={350}
    />
  );
}
