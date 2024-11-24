import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';
import { jwtDecode } from 'jwt-decode'; // Correct import
import logo from '../logo.png';

const Login = ({ setIsAuthenticated ,setIsAdmin}) => {
  const [user_email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await api.post('/login', { email: user_email, password });
      
      if (response && response.data) {
        // Decode the token to extract user information
        const decodedToken = jwtDecode(response.data.access_token);
        const { id, email, is_admin } = decodedToken; // Destructure the decoded token
        
        // Store token and other user info in localStorage
        localStorage.setItem('token', response.data.access_token);
        localStorage.setItem('id', id); // Store user ID
        localStorage.setItem('email', email); // Store email
        localStorage.setItem('is_admin', is_admin); // Store admin status
        setIsAuthenticated(true);
        setIsAdmin(is_admin);
        navigate('/dashboard');
      } else {
        throw new Error("Response data is missing");
      }
    } catch (error) {
      console.error("Login error:", error);
      alert("Login error: " + (error.response?.data?.detail || error.message));
    }
  };
  

  return (
    <div className="flex flex-col justify-center items-center h-screen">
      <form className="bg-white border border-gray-300 p-8 shadow-md rounded-lg w-96" onSubmit={handleSubmit}>

        <div className="flex flex-col justify-center items-center">
          <img src={logo} alt="Logo" className="mb-4" />
        </div>
        <h2 className="text-2xl mb-6 text-center">Synergy Budget Analysis</h2>
        <input
          type="email"
          placeholder="Email"
          value={user_email}
          onChange={(e) => setEmail(e.target.value)}
          className="border p-2 mb-4 w-full"
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="border p-2 mb-4 w-full"
        />
        <div className="flex flex-col justify-center items-center">
          <button type="submit" className="bg-blue-900 text-white px-10 py-2 w-70%">SIGN IN</button>
        </div>
      </form>
    </div>

  );
};

export default Login;
