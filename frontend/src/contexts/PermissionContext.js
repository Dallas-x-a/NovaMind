import React, { createContext, useContext, useState, useEffect } from 'react';
import { useAuth } from './AuthContext';

const PermissionContext = createContext();

export const usePermissions = () => {
  const context = useContext(PermissionContext);
  if (!context) {
    throw new Error('usePermissions must be used within a PermissionProvider');
  }
  return context;
};

export const PermissionProvider = ({ children }) => {
  const { user } = useAuth();
  const [permissions, setPermissions] = useState([]);
  const [roles, setRoles] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      fetchUserPermissions();
    } else {
      setPermissions([]);
      setRoles([]);
      setLoading(false);
    }
  }, [user]);

  const fetchUserPermissions = async () => {
    try {
      const response = await fetch('/api/users/permissions', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setPermissions(data.permissions || []);
        setRoles(data.roles || []);
      }
    } catch (error) {
      console.error('Error fetching permissions:', error);
    } finally {
      setLoading(false);
    }
  };

  const hasPermission = (permission) => {
    if (!user || !permissions.length) return false;
    
    // Check if user has the specific permission
    return permissions.includes(permission);
  };

  const hasAnyPermission = (permissionList) => {
    if (!user || !permissions.length) return false;
    
    return permissionList.some(permission => permissions.includes(permission));
  };

  const hasAllPermissions = (permissionList) => {
    if (!user || !permissions.length) return false;
    
    return permissionList.every(permission => permissions.includes(permission));
  };

  const hasRole = (role) => {
    if (!user || !roles.length) return false;
    
    return roles.includes(role);
  };

  const value = {
    permissions,
    roles,
    loading,
    hasPermission,
    hasAnyPermission,
    hasAllPermissions,
    hasRole,
    refreshPermissions: fetchUserPermissions
  };

  return (
    <PermissionContext.Provider value={value}>
      {children}
    </PermissionContext.Provider>
  );
}; 