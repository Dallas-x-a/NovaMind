import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { PermissionProvider } from './contexts/PermissionContext';

// Layout Components
import Layout from './components/Layout/Layout';
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';

// Page Components
import Dashboard from './pages/Dashboard/Dashboard';
import Agents from './pages/Agents/Agents';
import Tasks from './pages/Tasks/Tasks';
import Knowledge from './pages/Knowledge/Knowledge';
import Projects from './pages/Projects/Projects';
import Users from './pages/Users/Users';
import Settings from './pages/Settings/Settings';
import Login from './pages/Auth/Login';
import Register from './pages/Auth/Register';
import Profile from './pages/Profile/Profile';

// Permission Components
import ProtectedRoute from './components/Auth/ProtectedRoute';
import PermissionGuard from './components/Auth/PermissionGuard';

// Theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
  components: {
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#fff',
          borderRight: '1px solid #e0e0e0',
        },
      },
    },
  },
});

function AppContent() {
  const { user, loading } = useAuth();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  if (loading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <div>Loading...</div>
      </Box>
    );
  }

  if (!user) {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    );
  }

  return (
    <Layout>
      <Sidebar open={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />
      <Box sx={{ display: 'flex', flexDirection: 'column', flexGrow: 1 }}>
        <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
        <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            
            {/* Dashboard */}
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <PermissionGuard permission="dashboard:view">
                    <Dashboard />
                  </PermissionGuard>
                </ProtectedRoute>
              }
            />

            {/* Agents Management */}
            <Route
              path="/agents"
              element={
                <ProtectedRoute>
                  <PermissionGuard permission="agents:view">
                    <Agents />
                  </PermissionGuard>
                </ProtectedRoute>
              }
            />
            <Route
              path="/agents/:id"
              element={
                <ProtectedRoute>
                  <PermissionGuard permission="agents:view">
                    <Agents />
                  </PermissionGuard>
                </ProtectedRoute>
              }
            />

            {/* Tasks Management */}
            <Route
              path="/tasks"
              element={
                <ProtectedRoute>
                  <PermissionGuard permission="tasks:view">
                    <Tasks />
                  </PermissionGuard>
                </ProtectedRoute>
              }
            />
            <Route
              path="/tasks/:id"
              element={
                <ProtectedRoute>
                  <PermissionGuard permission="tasks:view">
                    <Tasks />
                  </PermissionGuard>
                </ProtectedRoute>
              }
            />

            {/* Knowledge Management */}
            <Route
              path="/knowledge"
              element={
                <ProtectedRoute>
                  <PermissionGuard permission="knowledge:view">
                    <Knowledge />
                  </PermissionGuard>
                </ProtectedRoute>
              }
            />
            <Route
              path="/knowledge/:id"
              element={
                <ProtectedRoute>
                  <PermissionGuard permission="knowledge:view">
                    <Knowledge />
                  </PermissionGuard>
                </ProtectedRoute>
              }
            />

            {/* Projects Management */}
            <Route
              path="/projects"
              element={
                <ProtectedRoute>
                  <PermissionGuard permission="projects:view">
                    <Projects />
                  </PermissionGuard>
                </ProtectedRoute>
              }
            />
            <Route
              path="/projects/:id"
              element={
                <ProtectedRoute>
                  <PermissionGuard permission="projects:view">
                    <Projects />
                  </PermissionGuard>
                </ProtectedRoute>
              }
            />

            {/* User Management */}
            <Route
              path="/users"
              element={
                <ProtectedRoute>
                  <PermissionGuard permission="users:view">
                    <Users />
                  </PermissionGuard>
                </ProtectedRoute>
              }
            />
            <Route
              path="/users/:id"
              element={
                <ProtectedRoute>
                  <PermissionGuard permission="users:view">
                    <Users />
                  </PermissionGuard>
                </ProtectedRoute>
              }
            />

            {/* Settings */}
            <Route
              path="/settings"
              element={
                <ProtectedRoute>
                  <PermissionGuard permission="settings:view">
                    <Settings />
                  </PermissionGuard>
                </ProtectedRoute>
              }
            />

            {/* Profile */}
            <Route
              path="/profile"
              element={
                <ProtectedRoute>
                  <Profile />
                </ProtectedRoute>
              }
            />

            {/* Catch all route */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Box>
      </Box>
    </Layout>
  );
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AuthProvider>
        <PermissionProvider>
          <Router>
            <AppContent />
          </Router>
        </PermissionProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App; 