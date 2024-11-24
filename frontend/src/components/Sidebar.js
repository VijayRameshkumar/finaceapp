import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';
import logo from '../logo.jpg';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faPowerOff,
  faTachometerAlt,
  faUser,
  faFileAlt,
  faChartPie,
  faCog,
  faUsers,
  faBars,
  faTimes,
  faUserCircle,
  faTh
} from '@fortawesome/free-solid-svg-icons';
import { getUserPermissions } from '../services/api';

const Sidebar = ({ setIsAuthenticated }) => {
  const [isAdmin, setIsAdmin] = useState(false);
  const [permissions, setPermissions] = useState({
    can_access_budget_analysis: false,
    can_access_accounts: false,
    can_access_analysis_report: false,
  });
  const [isCollapsed, setIsCollapsed] = useState(true);

  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = localStorage.getItem('token');
        const user_id = localStorage.getItem('id');
        if (token) {
          const response = await getUserPermissions(user_id);
          setIsAdmin(response.data.is_admin);
          setPermissions({
            can_access_budget_analysis: response.data.can_access_budget_analysis,
            can_access_accounts: response.data.can_access_accounts,
            can_access_analysis_report: response.data.can_access_analysis_report,
          });
        }
      } catch (error) {
        console.error("Error fetching user data:", error);
      }
    };

    fetchUserData();
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('id');
    localStorage.removeItem('email');
    localStorage.removeItem('is_admin');

    setIsAuthenticated(false);
    navigate('/login');
  };

  const linkClasses = (path) =>
    `flex items-center ${location.pathname === path ? 'text-Cyan-blue font-semibold' : 'hover:text-Cyan-blue'}`;

  return (
    <div className={`bg-blue-950 text-white w-${isCollapsed ? '16' : '64'} transition-width duration-300 min-h-screen p-4 flex flex-col justify-between`}>
      <div>
        <img className={`${isCollapsed ? 'block' : 'hidden'} pt-5`} src={logo} alt="Logo" width={30} />
        <h1 className={`text-xl font-bold text-white-200 ${isCollapsed ? 'hidden' : 'block'} mb-4`}>Synergy Budget Analysis</h1>

        <ul className="mt-4 item-center">
          <li className="mb-2 ml-2 mt-2 pt-1">
            <Link to="/dashboard" className={linkClasses('/dashboard')}>
              {isCollapsed ? <FontAwesomeIcon icon={faTachometerAlt} /> : <><FontAwesomeIcon icon={faTachometerAlt} className="mr-2 " /> Dashboard</>}
            </Link>
          </li>

          {permissions.can_access_budget_analysis && (
            <li className="mb-2 ml-2 mt-2 pt-1">
              <Link to="/budget-analysis" className={linkClasses('/budget-analysis')}>
                {isCollapsed ? <FontAwesomeIcon icon={faChartPie} /> : <><FontAwesomeIcon icon={faChartPie} className="mr-2" /> Budget Analysis</>}
              </Link>
            </li>
          )}

          {permissions.can_access_accounts && (
            <li className="mb-2 ml-2 pt-1">
              <Link to="/accounts" className={linkClasses('/accounts')}>
                {isCollapsed ? <FontAwesomeIcon icon={faUsers} /> : <><FontAwesomeIcon icon={faUsers} className="mr-2" /> Accounts</>}
              </Link>
            </li>
          )}

          {permissions.can_access_analysis_report && (
            <li className="mb-2 ml-3 pt-1">
              <Link to="/analysis-report" className={linkClasses('/analysis-report')}>
                {isCollapsed ? <FontAwesomeIcon icon={faFileAlt} /> : <><FontAwesomeIcon icon={faFileAlt} className="mr-2" /> Analysis Report</>}
              </Link>
            </li>
          )}
        </ul>
      </div>

      <div>
        <ul className="mt-4 item-center">
          {isAdmin && (
            <li className="mb-2 ml-2 pt-1">
              <Link to="/admin-console" className={linkClasses('/admin-console')}>
                {isCollapsed ? <FontAwesomeIcon icon={faCog} /> : <><FontAwesomeIcon icon={faCog} className="mr-2" /> Admin Console</>}
              </Link>
            </li>
          )}

          <li className="mb-2 ml-2 pt-1">
            <Link to="/profile" className={linkClasses('/profile')}>
              {isCollapsed ? <FontAwesomeIcon icon={faUserCircle} /> : <><FontAwesomeIcon icon={faUserCircle} className="mr-2" /> Profile</>}
            </Link>
          </li>
        </ul>
        <button onClick={handleLogout} className="hover:text-red-800 text-red-500 flex items-center mb-4 pt-2">
          <FontAwesomeIcon icon={faPowerOff} className="mr-2 ml-2" />
          {!isCollapsed && 'Logout'}
        </button>
        {/* Horizontal Divider */}
        <hr className="my-6 border-gray-100" />
        <button onClick={() => setIsCollapsed(!isCollapsed)} className="text-white mb-4">
          <FontAwesomeIcon icon={isCollapsed ? faBars : faTimes} className="mr-2 ml-2" />
        </button>
      </div>
    </div>
  );
};

export default Sidebar;
