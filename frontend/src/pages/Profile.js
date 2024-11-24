import React, { useState, useEffect } from 'react';
import {resetPassword } from '../services/api';
 
const Profile = () => {
  const [newPassword, setResetPassword] = useState('');
  const [userId, setId] = useState('');

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = localStorage.getItem('token');
        const id = localStorage.getItem('id');
  
        if (token) {
          setId(id);
        }
      } catch (error) {
        console.error("Error fetching user data:", error);
      }
    };

    fetchUserData();
  }, []);



  const handleResetPassword = async (e) => {
    e.preventDefault();

      // Check if newPassword is empty
  if (!newPassword) {
    alert('Error: Password cannot be empty.');
    return; // Exit the function if newPassword is empty
  }

    try {
      await resetPassword(userId, newPassword);
      alert('Reset was successful.');
      setResetPassword('');

    } catch (error) {
      console.error(error);
      alert('Error.');
    }
  };

  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold my-4">Profile</h2>

      {/* Create User and Delete User Forms */}
      <div className="flex justify-between md-6">
        <div className="flex flex-col md-6">
        
              <h3 className="font-semibold">Reset your Password</h3>
              <input
                type="test"
                value={newPassword}
                onChange={(e) => setResetPassword(e.target.value)}
                placeholder="New Password"
                required
                className="border p-2 mb-2"
              />
              <button type="submit" onClick={handleResetPassword} className="bg-blue-500 text-white px-4 py-2 rounded">Reset</button>
      
            {/* Update User Permissions */}
            <hr className="my-6 border-gray-500" />
          </div>
 
      </div>
    </div>
  );
};

export default Profile;
