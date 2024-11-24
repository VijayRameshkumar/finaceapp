import React from 'react';
import Sidebar from './Sidebar'; // Assuming Sidebar is in the same directory

// components/Layout.js
const Layout = ({ children, setIsAuthenticated }) => {
  return (
    <div className="flex">
      <Sidebar setIsAuthenticated={setIsAuthenticated} />
      <div className="flex-grow p-4">
        {children} {/* This is where your page components will render */}
      </div>
    </div>
  );
};


export default Layout;
