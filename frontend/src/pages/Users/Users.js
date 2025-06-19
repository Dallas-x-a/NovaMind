import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Button,
  IconButton,
  Chip,
  Typography,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  Grid,
  Card,
  CardContent,
  Avatar,
  Tooltip,
  Alert,
  Snackbar
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Security as SecurityIcon,
  Person as PersonIcon,
  AdminPanelSettings as AdminIcon,
  VerifiedUser as VerifiedIcon
} from '@mui/icons-material';
import { useAuth } from '../../contexts/AuthContext';
import { usePermissions } from '../../contexts/PermissionContext';
import UserForm from './components/UserForm';
import RoleForm from './components/RoleForm';
import PermissionMatrix from './components/PermissionMatrix';

const Users = () => {
  const { user: currentUser } = useAuth();
  const { hasPermission } = usePermissions();
  
  const [users, setUsers] = useState([]);
  const [roles, setRoles] = useState([]);
  const [permissions, setPermissions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [selectedUser, setSelectedUser] = useState(null);
  const [selectedRole, setSelectedRole] = useState(null);
  
  // Dialog states
  const [userDialogOpen, setUserDialogOpen] = useState(false);
  const [roleDialogOpen, setRoleDialogOpen] = useState(false);
  const [permissionDialogOpen, setPermissionDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  
  // Form states
  const [userForm, setUserForm] = useState({
    username: '',
    email: '',
    firstName: '',
    lastName: '',
    role: '',
    status: 'active',
    password: ''
  });
  
  const [roleForm, setRoleForm] = useState({
    name: '',
    description: '',
    permissions: []
  });
  
  // Notification states
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success'
  });

  useEffect(() => {
    fetchUsers();
    fetchRoles();
    fetchPermissions();
  }, []);

  const fetchUsers = async () => {
    try {
      const response = await fetch('/api/users', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      const data = await response.json();
      setUsers(data);
    } catch (error) {
      console.error('Error fetching users:', error);
      showSnackbar('Failed to fetch users', 'error');
    } finally {
      setLoading(false);
    }
  };

  const fetchRoles = async () => {
    try {
      const response = await fetch('/api/roles', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      const data = await response.json();
      setRoles(data);
    } catch (error) {
      console.error('Error fetching roles:', error);
    }
  };

  const fetchPermissions = async () => {
    try {
      const response = await fetch('/api/permissions', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      const data = await response.json();
      setPermissions(data);
    } catch (error) {
      console.error('Error fetching permissions:', error);
    }
  };

  const handleCreateUser = async (userData) => {
    try {
      const response = await fetch('/api/users', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(userData)
      });
      
      if (response.ok) {
        showSnackbar('User created successfully', 'success');
        setUserDialogOpen(false);
        fetchUsers();
      } else {
        const error = await response.json();
        showSnackbar(error.message || 'Failed to create user', 'error');
      }
    } catch (error) {
      console.error('Error creating user:', error);
      showSnackbar('Failed to create user', 'error');
    }
  };

  const handleUpdateUser = async (userId, userData) => {
    try {
      const response = await fetch(`/api/users/${userId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(userData)
      });
      
      if (response.ok) {
        showSnackbar('User updated successfully', 'success');
        setUserDialogOpen(false);
        fetchUsers();
      } else {
        const error = await response.json();
        showSnackbar(error.message || 'Failed to update user', 'error');
      }
    } catch (error) {
      console.error('Error updating user:', error);
      showSnackbar('Failed to update user', 'error');
    }
  };

  const handleDeleteUser = async (userId) => {
    try {
      const response = await fetch(`/api/users/${userId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        showSnackbar('User deleted successfully', 'success');
        setDeleteDialogOpen(false);
        fetchUsers();
      } else {
        const error = await response.json();
        showSnackbar(error.message || 'Failed to delete user', 'error');
      }
    } catch (error) {
      console.error('Error deleting user:', error);
      showSnackbar('Failed to delete user', 'error');
    }
  };

  const handleCreateRole = async (roleData) => {
    try {
      const response = await fetch('/api/roles', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(roleData)
      });
      
      if (response.ok) {
        showSnackbar('Role created successfully', 'success');
        setRoleDialogOpen(false);
        fetchRoles();
      } else {
        const error = await response.json();
        showSnackbar(error.message || 'Failed to create role', 'error');
      }
    } catch (error) {
      console.error('Error creating role:', error);
      showSnackbar('Failed to create role', 'error');
    }
  };

  const handleUpdateRole = async (roleId, roleData) => {
    try {
      const response = await fetch(`/api/roles/${roleId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(roleData)
      });
      
      if (response.ok) {
        showSnackbar('Role updated successfully', 'success');
        setRoleDialogOpen(false);
        fetchRoles();
      } else {
        const error = await response.json();
        showSnackbar(error.message || 'Failed to update role', 'error');
      }
    } catch (error) {
      console.error('Error updating role:', error);
      showSnackbar('Failed to update role', 'error');
    }
  };

  const showSnackbar = (message, severity = 'success') => {
    setSnackbar({
      open: true,
      message,
      severity
    });
  };

  const handleCloseSnackbar = () => {
    setSnackbar({
      ...snackbar,
      open: false
    });
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active':
        return 'success';
      case 'inactive':
        return 'error';
      case 'pending':
        return 'warning';
      default:
        return 'default';
    }
  };

  const getRoleIcon = (roleName) => {
    switch (roleName.toLowerCase()) {
      case 'admin':
        return <AdminIcon />;
      case 'user':
        return <PersonIcon />;
      case 'moderator':
        return <VerifiedIcon />;
      default:
        return <PersonIcon />;
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <Typography>Loading users...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          User Management
        </Typography>
        <Box>
          {hasPermission('users:create') && (
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => {
                setSelectedUser(null);
                setUserForm({
                  username: '',
                  email: '',
                  firstName: '',
                  lastName: '',
                  role: '',
                  status: 'active',
                  password: ''
                });
                setUserDialogOpen(true);
              }}
              sx={{ mr: 1 }}
            >
              Add User
            </Button>
          )}
          {hasPermission('roles:create') && (
            <Button
              variant="outlined"
              startIcon={<SecurityIcon />}
              onClick={() => {
                setSelectedRole(null);
                setRoleForm({
                  name: '',
                  description: '',
                  permissions: []
                });
                setRoleDialogOpen(true);
              }}
            >
              Manage Roles
            </Button>
          )}
        </Box>
      </Box>

      {/* Statistics Cards */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
                  <PersonIcon />
                </Avatar>
                <Box>
                  <Typography variant="h6">{users.length}</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Total Users
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Avatar sx={{ bgcolor: 'success.main', mr: 2 }}>
                  <VerifiedIcon />
                </Avatar>
                <Box>
                  <Typography variant="h6">
                    {users.filter(u => u.status === 'active').length}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Active Users
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Avatar sx={{ bgcolor: 'warning.main', mr: 2 }}>
                  <AdminIcon />
                </Avatar>
                <Box>
                  <Typography variant="h6">{roles.length}</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Total Roles
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Avatar sx={{ bgcolor: 'info.main', mr: 2 }}>
                  <SecurityIcon />
                </Avatar>
                <Box>
                  <Typography variant="h6">{permissions.length}</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Total Permissions
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Users Table */}
      <Paper>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>User</TableCell>
                <TableCell>Email</TableCell>
                <TableCell>Role</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Last Login</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {users
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((user) => (
                  <TableRow key={user.id}>
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        <Avatar sx={{ mr: 2 }}>
                          {user.firstName?.[0] || user.username[0]}
                        </Avatar>
                        <Box>
                          <Typography variant="body1">
                            {user.firstName} {user.lastName}
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            @{user.username}
                          </Typography>
                        </Box>
                      </Box>
                    </TableCell>
                    <TableCell>{user.email}</TableCell>
                    <TableCell>
                      <Chip
                        icon={getRoleIcon(user.role)}
                        label={user.role}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={user.status}
                        color={getStatusColor(user.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      {user.lastLogin
                        ? new Date(user.lastLogin).toLocaleDateString()
                        : 'Never'}
                    </TableCell>
                    <TableCell>
                      <Box>
                        {hasPermission('users:update') && (
                          <Tooltip title="Edit User">
                            <IconButton
                              size="small"
                              onClick={() => {
                                setSelectedUser(user);
                                setUserForm({
                                  username: user.username,
                                  email: user.email,
                                  firstName: user.firstName || '',
                                  lastName: user.lastName || '',
                                  role: user.role,
                                  status: user.status,
                                  password: ''
                                });
                                setUserDialogOpen(true);
                              }}
                            >
                              <EditIcon />
                            </IconButton>
                          </Tooltip>
                        )}
                        {hasPermission('users:delete') && user.id !== currentUser.id && (
                          <Tooltip title="Delete User">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => {
                                setSelectedUser(user);
                                setDeleteDialogOpen(true);
                              }}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                        )}
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[5, 10, 25]}
          component="div"
          count={users.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={(event, newPage) => setPage(newPage)}
          onRowsPerPageChange={(event) => {
            setRowsPerPage(parseInt(event.target.value, 10));
            setPage(0);
          }}
        />
      </Paper>

      {/* User Dialog */}
      <UserForm
        open={userDialogOpen}
        onClose={() => setUserDialogOpen(false)}
        user={selectedUser}
        roles={roles}
        onSubmit={selectedUser ? handleUpdateUser : handleCreateUser}
      />

      {/* Role Dialog */}
      <RoleForm
        open={roleDialogOpen}
        onClose={() => setRoleDialogOpen(false)}
        role={selectedRole}
        permissions={permissions}
        onSubmit={selectedRole ? handleUpdateRole : handleCreateRole}
      />

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete User</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete user "{selectedUser?.username}"? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => handleDeleteUser(selectedUser?.id)}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
      >
        <Alert
          onClose={handleCloseSnackbar}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Users; 