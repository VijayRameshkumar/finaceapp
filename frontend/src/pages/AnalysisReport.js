import React, { useState, useEffect } from 'react';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';
import api from '../services/api';
 

const AnalysisReport = () => {
 
  const [permissions, setPermissions] = useState({
    can_access_analysis_report: false,
    can_download: false,
  });
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [pdfurl, setPdfurl] = useState('https://drive.google.com/file/d/1Q_H5GOf_vItuE-OqHaeLhvRfQ8tvchnf/preview');

  const pdf_fileId = '1Q_H5GOf_vItuE-OqHaeLhvRfQ8tvchnf';
  const pdf_downloadUrl = `https://drive.google.com/uc?id=${pdf_fileId}&export=download`;

  const xlsx_fileId = '2PACX-1vRQcIJl_fvoKRR8Ke3EUP9psxIjj3Z7lyv6WmAaGd1E2uUpxuwLdiboeikVEx5ZVw';
  const xlsx_downloadUrl = `https://docs.google.com/spreadsheets/d/1iXlrPlE3p4svjFvURwIbqfPMGm3iekT3/export?format=xlsx`;
 
  const refeshdata = () => {
    setPdfurl('Drewry.pdf')
  };

  const fetchUserData = async () => {
    try {
      const token = localStorage.getItem('token');
      const user_id = localStorage.getItem('id');
      if (token) {
        const response = await api.get(`/user_permissions/${user_id}`);
        setPermissions({
          can_access_budget_analysis: response.data.can_access_budget_analysis,
          can_access_accounts: response.data.can_access_accounts,
          can_access_analysis_report: response.data.can_access_analysis_report,
          can_download: response.data.can_download,
        });
      }
    } catch (error) {
      console.error("Error fetching user data:", error);
    }
  };

  useEffect(() => {
    fetchUserData();
  }, []);

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold">Analysis Report</h2>
        <button
          onClick={refeshdata}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Refresh
        </button>
      </div>

      <Tabs selectedIndex={selectedIndex} onSelect={index => setSelectedIndex(index)}>
        <TabList style={{ backgroundColor: 'white', boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.2)' }}>
          <Tab style={{ backgroundColor: selectedIndex === 0 ? '#041085' : 'white', color: selectedIndex === 0 ? 'white' : 'black', boxShadow: selectedIndex === 0 ? '0px 4px 10px rgba(0, 0, 0, 0.2)' : 'none' }}>
          Yearly Report
          </Tab>
          <Tab style={{ backgroundColor: selectedIndex === 1 ? '#041085' : 'white', color: selectedIndex === 1 ? 'white' : 'black', boxShadow: selectedIndex === 0 ? '0px 4px 10px rgba(0, 0, 0, 0.2)' : 'none' }}>
          Analysis Report
          </Tab>
        </TabList>

        <TabPanel>
          <h3>Yearly - PDF Report</h3>
          {permissions.can_access_analysis_report ? (
            <div>
              <iframe
                src={pdfurl}
                allow="autoplay"
                width="100%"
                height="500"
                style={{
                  border: '1px solid #000',
                  borderRadius: '3px',
                  boxShadow: '0 2px 5px rgba(0, 0, 0, 0.3)'
                }}
              ></iframe>
              {permissions.can_download && (
                <a
                  href={pdf_downloadUrl}
                  download
                  className="mt-4 inline-block bg-blue-500 text-white px-4 py-2 rounded"
                >
                  Download
                </a>
              )}
            </div>
          ) : (
            <p>Loading...</p>
          )}
        </TabPanel>

        <TabPanel>
          <h3>Analysis - Excel Report</h3>
          {permissions.can_access_analysis_report ? (
            <div>
              <iframe
                src={`https://docs.google.com/spreadsheets/d/e/${xlsx_fileId}/pubhtml?widget=true&headers=false`}
                width="100%"
                height="500"
                style={{
                  border: '1px solid #000',
                  borderRadius: '3px',
                  boxShadow: '0 2px 5px rgba(0, 0, 0, 0.3)'
                }}
              ></iframe>
              {permissions.can_download && (
                <a
                  href={xlsx_downloadUrl}
                  download
                  className="mt-4 inline-block bg-blue-500 text-white px-4 py-2 rounded"
                >
                  Download
                </a>
              )}
            </div>
          ) : (
            <p>Loading...</p>
          )}
        </TabPanel>
      </Tabs>
    </div>
  );
};

export default AnalysisReport;
