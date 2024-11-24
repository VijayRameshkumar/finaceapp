import React, { useState, useEffect } from 'react';
import { createUser, deleteUser, updateUserPermissions, getUserPermissions, getUsers } from '../services/api';
import Table from '../components/Table'; // Import the Table component
import SearchIcon from '@mui/icons-material/Search';
import Box from '@mui/material/Box'; 

const AdminConsole = () => {
  const [userEmail, setUserEmail] = useState('');
  const [userName, setUserName] = useState('');
  const [userIdToDelete, setUserIdToDelete] = useState('');
  const [userIdToUpdate, setUserIdToUpdate] = useState('');
  const [userIdToFetch, setUserIdToFetch] = useState('');
  const [searchEmail, setSearchEmail] = useState('');
  const [userPassword, setUserPassword] = useState('');
  const [permissions, setPermissions] = useState({
    can_access_budget_analysis: false,
    can_access_accounts: false,
    can_access_analysis_report: false,
    can_download: false,
  });
  const [fetchedUser, setFetchedUser] = useState(null);
  const [allUsers, setAllUsers] = useState([]);

  const fetchUsers = async (searchTerm) => {
    try {
      const response = await getUsers(searchTerm); // Pass null or empty string to your API
      setAllUsers(response.data);
    } catch (error) {
      console.error('Error fetching users:', error);
    }
  };

  // Fetch users on mount and whenever searchEmail changes
  useEffect(() => {
    const debounceFetch = setTimeout(() => {
      fetchUsers(searchEmail.trim() === '' ? null : searchEmail); // Pass null to fetch all users
    }, 300); // Adjust the debounce time as needed

    return () => clearTimeout(debounceFetch);
  }, [searchEmail]);

  const searchHandleChange = (value) => {
    setSearchEmail(value);
  };

  // Create User
  const handleCreateUser = async (e) => {
    e.preventDefault();
    try {
      await createUser({ email: userEmail, name: userName, is_admin: false, password: userPassword });
      alert('User created successfully and Sent Email.');
      setUserEmail('');
      setUserName('');
      setUserPassword('');
      fetchUsers(searchEmail.trim() === '' ? null : searchEmail); // Re-fetch users to reflect the new user
    } catch (error) {
      console.error(error);
      alert('Error creating user');
    }
  };

  // Delete User
  const handleDeleteUser = async (userId) => {
    try {
      await deleteUser(userId);
      alert('User deleted successfully');
      setUserIdToDelete('');
      fetchUsers(searchEmail.trim() === '' ? null : searchEmail); // Re-fetch users
    } catch (error) {
      console.error(error);
      alert('Error deleting user');
    }
  };

  // Update User Permissions
  const handleUpdatePermissions = async () => {
    try {
      await updateUserPermissions(userIdToUpdate, permissions);
      alert('Permissions updated successfully');
      setUserIdToUpdate('');
      setPermissions({
        can_access_budget_analysis: false,
        can_access_accounts: false,
        can_access_analysis_report: false,
        can_download: false,
      });
      fetchUsers(searchEmail.trim() === '' ? null : searchEmail); // Re-fetch users
    } catch (error) {
      console.error(error);
      alert('Error updating permissions');
    }
  };

  // Fetch User Permissions
  const handleFetchPermissions = async () => {
    try {
      const response = await getUserPermissions(userIdToFetch);
      setFetchedUser(response.data);
    } catch (error) {
      console.error(error);
      setFetchedUser(null);
      alert('Error fetching user permissions');
    }
  };

  // Prepare fetched user table data
  const fetchedUserTableHeaders = ['Field', 'Value'];
  const fetchedUserTableData = fetchedUser
    ? [
      ['Email', fetchedUser.email],
      ['Name', fetchedUser.name],
      ['Is Admin', fetchedUser.is_admin ? 'Yes' : 'No'],
      ['Can Access Budget Analysis', fetchedUser.can_access_budget_analysis ? 'Yes' : 'No'],
      ['Can Access Accounts', fetchedUser.can_access_accounts ? 'Yes' : 'No'],
      ['Can Access Analysis Report', fetchedUser.can_access_analysis_report ? 'Yes' : 'No'],
      ['Can Download', fetchedUser.can_download ? 'Yes' : 'No'],
    ]
    : [];

  // Prepare all users table data
  const allUsersTableHeaders = [
    'ID', 'Email', 'Name', 'Is Admin', 'Can Access Budget Analysis', 'Can Access Accounts', 'Can Access Analysis Report', 'Can Download',
  ];
  const allUsersTableData = allUsers.map((user) => [
    user.id,
    user.email,
    user.name,
    user.is_admin ? 'Yes' : 'No',
    user.can_access_budget_analysis ? 'Yes' : 'No',
    user.can_access_accounts ? 'Yes' : 'No',
    user.can_access_analysis_report ? 'Yes' : 'No',
    user.can_download ? 'Yes' : 'No',
  ]);

  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold my-4">Admin Console</h2>

      {/* Create User and Delete User Forms */}
      <div className="flex justify-between sm-12">

        <div className="flex flex-col">
          <h3 className="font-semibold">Create User</h3>
          <input
            type="email"
            value={userEmail}
            onChange={(e) => setUserEmail(e.target.value)}
            placeholder="Email"
            required
            className="border p-2 mb-2 w-96"
          />
          <input
            type="text"
            value={userName}
            onChange={(e) => setUserName(e.target.value)}
            placeholder="Name"
            required
            className="border p-2 mb-2 w-96"
          />
          <input
            type="password"
            value={userPassword}
            onChange={(e) => setUserPassword(e.target.value)}
            placeholder="Password"
            required
            className="border p-2 mb-2 w-96"
          />
          <button type="submit" onClick={handleCreateUser} className="bg-blue-500 text-white px-4 py-2 rounded">Create User</button>

          {/* Update User Permissions */}

          {/* <div className="flex flex-col sm-4">
              <h3 className="font-semibold">Delete User</h3>
              <input
                type="text"
                value={userIdToDelete}
                onChange={(e) => setUserIdToDelete(e.target.value)}
                placeholder="User ID"
                required
                className="border p-2 mb-2"
              />
              <button onClick={handleDeleteUser} className="bg-red-500 text-white px-4 py-2 rounded">Delete User</button>
            </div> */}

          {/* Update User Permissions */}
          {/* <hr className="my-6 border-gray-500" /> */}

          {/* Fetch User Permissions */}
          {/* <hr className="my-6 border-gray-500" /> */}

          {/* Fetch User Permissions */}
          {/* <div className="sm-4">
            <h3 className="font-semibold">Fetch User Permissions</h3>
            <input
              type="text"
              value={userIdToFetch}
              onChange={(e) => setUserIdToFetch(e.target.value)}
              placeholder="User ID"
              required
              className="border p-2 mt-2 mr-2"
            />
            <button onClick={handleFetchPermissions} className="bg-blue-500 text-white px-4 py-2 rounded">Fetch Permissions</button>
            {fetchedUser ? (
              <div className="mt-4">
                <h4 className="font-semibold">User Details - Permissions:</h4>
                <Table headers={fetchedUserTableHeaders} data={fetchedUserTableData} />
              </div>
            ) : (
              <div className="flex flex-col justify-center items-center pt-24">
              </div>
            )}
          </div> */}
        </div>
        <div className="flex flex-col">
          <h3 className="font-semibold">Update User Permissions</h3>
          <input
            type="text"
            value={userIdToUpdate}
            onChange={(e) => setUserIdToUpdate(e.target.value)}
            placeholder="User ID"
            required
            className="border p-2 mb-2"
          />
          <div className="flex flex-col">
            <div>
              <label className="inline-flex items-center mx-4">
                <input
                  type="checkbox"
                  checked={permissions.can_access_budget_analysis}
                  onChange={(e) => setPermissions({ ...permissions, can_access_budget_analysis: e.target.checked })}
                  className="mr-1 mb-0.5"
                />
                Can Access Budget Analysis
              </label>
              <label className="inline-flex items-center mx-4">
                <input
                  type="checkbox"
                  checked={permissions.can_access_accounts}
                  onChange={(e) => setPermissions({ ...permissions, can_access_accounts: e.target.checked })}
                  className="mr-1 mb-0.5"
                />
                Can Access Accounts
              </label>
            </div>
            <div>
              <label className="inline-flex items-center mx-4">
                <input
                  type="checkbox"
                  checked={permissions.can_access_analysis_report}
                  onChange={(e) => setPermissions({ ...permissions, can_access_analysis_report: e.target.checked })}
                  className="mr-1 mb-0.5"
                />
                Can Access Analysis Report
              </label>
              <label className="inline-flex items-center mx-4">
                <input
                  type="checkbox"
                  checked={permissions.can_download}
                  onChange={(e) => setPermissions({ ...permissions, can_download: e.target.checked })}
                  className="mr-1 mb-0.5"
                />
                Can Download
              </label>
            </div>
          </div>
          <button onClick={handleUpdatePermissions} className="bg-blue-500 text-white px-4 py-2 rounded mt-12">Update Permissions</button>
        </div>

        {/* Search Users */}
      </div>
      <hr className="my-6 border-gray-500" />
      <Box sx={{
        backgroundColor: 'white',
        borderRadius: 2,
        boxShadow: 2,
        padding: 2,
      }}>
      <div className="flex flex-col sm-4">
        <h3 className="font-semibold">All Users</h3>
        <div className="relative mb-2">
          <input
            type="text"
            value={searchEmail}
            onChange={(e) => searchHandleChange(e.target.value)}
            placeholder="Search by Email"
            className="border p-2 pl-10 mb-2" // Add padding to the left for the icon
          />
          <SearchIcon className="absolute left-2 top-2 text-gray-500" />
        </div>
        <Table headers={allUsersTableHeaders} data={allUsersTableData} onDelete={handleDeleteUser}/>
      </div>
      </Box>

    </div>
  );
};

export default AdminConsole;
