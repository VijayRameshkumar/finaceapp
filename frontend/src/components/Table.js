// src/components/Table.js
import React, { useState } from 'react';
import { Delete } from '@mui/icons-material'; // Import the Delete icon from Material-UI

const Table = ({ headers, data, onDelete }) => {
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 5; // Set how many rows to display per page

  // Calculate total pages
  const totalPages = Math.ceil(data.length / rowsPerPage);

  // Get current rows for the page
  const startIndex = (currentPage - 1) * rowsPerPage;
  const currentRows = data.slice(startIndex, startIndex + rowsPerPage);

  // Handle page change
  const handlePageChange = (direction) => {
    if (direction === 'next' && currentPage < totalPages) {
      setCurrentPage(currentPage + 1);
    } else if (direction === 'prev' && currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };

  return (
    <div>
      <table className="min-w-full border-collapse border border-gray-200 text-sm">
        <thead>
          <tr>
            {headers.map((header, index) => (
              <th key={index} className="border border-gray-300 p-2">{header}</th>
            ))}
            <th className="border border-gray-300 p-2">Actions</th> {/* New Actions header */}
          </tr>
        </thead>
        <tbody>
          {currentRows.map((row, index) => (
            <tr key={index}>
              {row.map((cell, cellIndex) => (
                <td key={cellIndex} className="border border-gray-300 p-2">{cell}</td>
              ))}
              <td className="border border-gray-300 p-2"> {/* Actions column */}
                <button onClick={() => onDelete(row[0])} className="text-red-500 hover:text-red-700">
                  <Delete />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Pagination controls */}
      <div className="flex justify-between mt-4">
        <button
          onClick={() => handlePageChange('prev')}
          disabled={currentPage === 1}
          className="bg-blue-600 p-2 rounded disabled:opacity-50"
        >
          Previous
        </button>
        <span>{`Page ${currentPage} of ${totalPages}`}</span>
        <button
          onClick={() => handlePageChange('next')}
          disabled={currentPage === totalPages}
          className="bg-blue-600 p-2 rounded disabled:opacity-50"
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default Table;
