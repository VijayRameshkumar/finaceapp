import React from 'react';
import { LineChart } from '@mui/x-charts/LineChart';
import Box from '@mui/material/Box';
export default function QuartilesLineChart({ data }) {
    if (!data || !data.q2 || !data.median || !data.q3 || !data.dates) {
        return <p className="text-red-500">Loading....</p>;
    }

    // Function to format large numbers to 'K' format
    const formatYAxisValue = (value) => {
        if (value >= 1000) {
            return `${(value / 1000).toFixed(1)}K`; // Convert to K format with one decimal
        }
        return value;
    };

    return (
        <div>
              <Box sx={{
            backgroundColor: 'white',
            borderRadius: 2,
            boxShadow: 2,
            padding: 2,
          }}>
            <LineChart
                width={850}
                height={400}
                series={[
                    { data: data.q2, label: '50% Population', color: 'red' },
                    { data: data.median, label: 'Median', color: 'orange' },
                    { data: data.q3, label: '75% Population', color: 'green' },
                ]}
                xAxis={[{
                    scaleType: 'point', 
                    data: data.dates, 
                    label: 'Date',
                    paddingRight: 10 // Optional: space between label and axis
                }]}
                yAxis={[{ 
                    label: '', 
                    position: 'left', 
                    labelStyle: { marginRight: 10 },  // Add margin to prevent overlap
                    paddingLeft: 10, // Adjust this for more space if needed
                    valueFormatter: formatYAxisValue // Format the Y-axis values
                }]}
                tooltip={{
                    show: true,
                    series: true,
                }}
            />
            </Box>
        </div>
    );
}
