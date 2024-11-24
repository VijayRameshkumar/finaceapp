import logo from '../logo.png';
import { useNavigate } from 'react-router-dom';

const Dashboard = () => {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate('/admin-console');
  };
  return (
    <div className="p-4 relative">
      <h2 className="text-2xl font-bold">Dashboard</h2>
      <div class="container">
        <div class="grid grid-cols-1 md:grid-cols-2">
          <div class="pt-20">
            <h1 className="text-4xl font-bold pb-5 text-blue-700">WELCOME </h1>
            <h4 className="text-2md font-bold pb-5">Synergy budget Analysis Tool</h4>
            <p>Empower your financial management with tools <br/>for Budget Analysis, Accounts insights, and downloadable<br/> Analysis Reports.</p>
            <div className="flex flex-col justify-left">
              <button   onClick={handleClick}  className="bg-blue-900 text-white mt-2 px-10 py-2 w-44 font-bold">Get Start</button>
            </div>
          </div>
          <div class="p-6">
            <img src={logo} alt="Logo" width={275} className="mb-4" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
