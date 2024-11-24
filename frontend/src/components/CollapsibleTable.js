import React, { useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Collapse,
  Box,
  Typography,
  Paper,
  Button,
  TableSortLabel,
  TablePagination
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';

import { KeyboardArrowDown, KeyboardArrowUp } from '@mui/icons-material';

const CollapsibleTable = ({ data, allow }) => {
  const [order, setOrder] = useState('asc');
  const [orderBy, setOrderBy] = useState('');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10); // Page size is set to 5

  const handleRequestSort = (property) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const downloadCSV = () => {
    const headers = ['Name', 'Median 50 Prec Population', 'Optimal 63 Prec Population', 'Top 75 Prec Population'];

    const rows = data.map(row => [
      row.Header,
      row.median_50perc_population ?? 'N/A',
      row.optimal_63perc_population ?? 'N/A',
      row.top_75perc_population ?? 'N/A',
    ]);

    const csvContent = [
      headers.join(','), // Header row
      ...rows.map(row => row.join(',')) // Data rows
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'table_data.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const sortedData = [...data].sort((a, b) => {
    const isAsc = order === 'asc';
    if (a[orderBy] < b[orderBy]) {
      return isAsc ? -1 : 1;
    }
    if (a[orderBy] > b[orderBy]) {
      return isAsc ? 1 : -1;
    }
    return 0;
  });

  const paginatedData = sortedData.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);

  return (
    <>
      {allow && (
    <Button 
    variant="contained" 
    onClick={downloadCSV} 
    sx={{ marginBottom: 2 }}
    startIcon={<DownloadIcon />} // Adds the icon to the left of the text
  >
    Download
  </Button>
      )}
      <TableContainer component={Paper}>
        <Table aria-label="collapsible table" sx={{ border: '1px solid gray' }}>
          <TableHead>
            <TableRow>
              <TableCell style={{ fontWeight: 'bold', backgroundColor: '#072175', border: '1px solid gray', padding: '0', width: '5px' }} />
              <TableCell style={{ fontWeight: 'bold', backgroundColor: '#072175', fontSize: '14px', border: '1px solid gray', color: 'white', padding: '10px 1px 1px 10px', width: '50px' }}>
                <TableSortLabel
                  active={orderBy === 'Header'}
                  direction={orderBy === 'Header' ? order : 'asc'}
                  onClick={() => handleRequestSort('Header')}
                  sx={{ color: orderBy === 'Header' ? 'white' : 'white', '& .MuiTableSortLabel-icon': { color: 'white' } }}
                >
                  Name
                </TableSortLabel>
              </TableCell>
              <TableCell style={{ fontWeight: 'bold', backgroundColor: '#072175', fontSize: '14px', border: '1px solid gray', color: 'white', padding: '10px 1px 1px 10px', width: '150px' }}>
                <TableSortLabel
                  active={orderBy === 'median_50perc_population'}
                  direction={orderBy === 'median_50perc_population' ? order : 'asc'}
                  onClick={() => handleRequestSort('median_50perc_population')}
                  sx={{ color: orderBy === 'median_50perc_population' ? 'white' : 'white', '& .MuiTableSortLabel-icon': { color: 'white' } }}
                >
                  Median 50 Prec Population
                </TableSortLabel>
              </TableCell>
              <TableCell style={{ fontWeight: 'bold', backgroundColor: '#072175', fontSize: '14px', border: '1px solid gray', color: 'white', padding: '10px 1px 1px 10px', width: '150px' }}>
                <TableSortLabel
                  active={orderBy === 'optimal_63perc_population'}
                  direction={orderBy === 'optimal_63perc_population' ? order : 'asc'}
                  onClick={() => handleRequestSort('optimal_63perc_population')}
                  sx={{ color: orderBy === 'optimal_63perc_population' ? 'white' : 'white', '& .MuiTableSortLabel-icon': { color: 'white' } }}
                >
                  Optimal 63 Prec Population
                </TableSortLabel>
              </TableCell>
              <TableCell style={{ fontWeight: 'bold', backgroundColor: '#072175', fontSize: '14px', border: '1px solid gray', color: 'white', padding: '10px 2px 1px 5px', width: '130px' }}>
                <TableSortLabel
                  active={orderBy === 'top_75perc_population'}
                  direction={orderBy === 'top_75perc_population' ? order : 'asc'}
                  onClick={() => handleRequestSort('top_75perc_population')}
                  sx={{ color: orderBy === 'top_75perc_population' ? 'white' : 'white', '& .MuiTableSortLabel-icon': { color: 'white' } }}
                >
                  Top 75 Prec Population
                </TableSortLabel>
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {paginatedData.map((row, index) => (
              <Row key={index} row={row} index={index} />
            ))}
          </TableBody>
        </Table>
        <TablePagination
          rowsPerPageOptions={[10, 20, 30]}
          component="div"
          count={data.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </TableContainer>
    </>
  );
};

const Row = ({ row, index }) => {
  const [open, setOpen] = useState(false);

  const backgroundColor = index % 2 === 0 ? 'white' : '#e6e7e8';

  return (
    <>
      <TableRow style={{ backgroundColor }}>
        <TableCell style={{ border: '1px solid gray' }}>
          <IconButton
            aria-label="expand row"
            size="small"
            onClick={() => setOpen(!open)}
          >
            {open ? <KeyboardArrowUp /> : <KeyboardArrowDown />}
          </IconButton>
        </TableCell>
        <TableCell style={{ fontSize: '14px', border: '1px solid gray', color: 'black', padding: '10px 1px 1px 10px', width: '150px' }}>{row.Header}</TableCell>
        <TableCell style={{ fontSize: '14px', border: '1px solid gray', color: 'black', padding: '10px 1px 1px 10px', width: '150px' }}>{row.median_50perc_population ?? 'N/A'}</TableCell>
        <TableCell style={{ fontSize: '14px', border: '1px solid gray', color: 'black', padding: '10px 1px 1px 10px', width: '150px' }}>{row.optimal_63perc_population ?? 'N/A'}</TableCell>
        <TableCell style={{ fontSize: '14px', border: '1px solid gray', color: 'black', padding: '10px 1px 1px 10px', width: '150px' }}>{row.top_75perc_population ?? 'N/A'}</TableCell>
      </TableRow>
      <TableRow>
        <TableCell style={{ paddingBottom: 0, paddingTop: 0, border: '1px solid gray' }} colSpan={7}>
          <Collapse in={open} timeout="auto" unmountOnExit>
            <Box margin={1}>
              <Typography variant="h6" gutterBottom component="div">
                Records
              </Typography>
              {row.records && row.records.length > 0 ? (
                <Table size="small" aria-label="records">
                  <TableHead>
                    <TableRow>
                      <TableCell style={{ fontWeight: 'bold', backgroundColor: '#e0e0e0', border: '1px solid gray' }}>Order</TableCell>
                      <TableCell style={{ fontWeight: 'bold', backgroundColor: '#e0e0e0', border: '1px solid gray' }}>Categories</TableCell>
                      <TableCell style={{ fontWeight: 'bold', backgroundColor: '#e0e0e0', border: '1px solid gray' }}>Median 50 Prec Population</TableCell>
                      <TableCell style={{ fontWeight: 'bold', backgroundColor: '#e0e0e0', border: '1px solid gray' }}>Optimal 63 Prec Population</TableCell>
                      <TableCell style={{ fontWeight: 'bold', backgroundColor: '#e0e0e0', border: '1px solid gray' }}>Top 75 Prec Population</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {row.records.map((record, idx) => (
                      <TableRow key={idx}>
                        <TableCell style={{ border: '1px solid gray' }}>{record.order}</TableCell>
                        <TableCell style={{ border: '1px solid gray' }}>{record.CATEGORIES}</TableCell>
                        <TableCell style={{ border: '1px solid gray' }}>{record.median_50perc_population}</TableCell>
                        <TableCell style={{ border: '1px solid gray' }}>{record.optimal_63perc_population}</TableCell>
                        <TableCell style={{ border: '1px solid gray' }}>{record.top_75perc_population}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <Typography>No records available</Typography>
              )}
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
};

export default CollapsibleTable;
