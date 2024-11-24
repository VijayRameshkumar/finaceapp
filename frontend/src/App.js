import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Dashboard from './pages/Dashboard';
import AdminConsole from './pages/AdminConsole';
import BudgetAnalysis from './pages/BudgetAnalysis';
import Accounts from './pages/Accounts';
import AnalysisReport from './pages/AnalysisReport';
import Login from './pages/Login';
import Layout from './components/Layout';
import Profile from './pages/Profile';

const PrivateRoute = ({ element, isAdmin }) => {
  return isAdmin ? element : <Navigate to="/dashboard" />;
};

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('token'));
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = localStorage.getItem('token');
        const is_admin = localStorage.getItem('is_admin');
        if (token) {
          setIsAdmin(is_admin === 'true'); // Ensure is_admin is properly checked
        }
      } catch (error) {
        console.error("Error fetching user data:", error);
      }
    };

    fetchUserData();
  }, []);

  return (
    <Router>
      {isAuthenticated ? (
        <Layout setIsAuthenticated={setIsAuthenticated}>
          <Routes>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/budget-analysis" element={<BudgetAnalysis />} />
            <Route path="/accounts" element={<Accounts />} />
            <Route path="/analysis-report" element={<AnalysisReport />} />
            <Route path="/profile" element={<Profile />} />
            {/* Admin Routes */}
            <Route 
              path="/admin-console" 
              element={<PrivateRoute element={<AdminConsole />} isAdmin={isAdmin} />} 
            />
       

            {/* Redirect for unmatched routes */}
            <Route path="*" element={<Navigate to="/dashboard" />} />
          </Routes>
        </Layout>
      ) : (
        <Routes>
          <Route path="/login" element={<Login setIsAuthenticated={setIsAuthenticated} setIsAdmin={setIsAdmin}/>} />
          <Route path="*" element={<Navigate to="/login" />} />
        </Routes>
      )}
    </Router>
  );
};

export default App;
