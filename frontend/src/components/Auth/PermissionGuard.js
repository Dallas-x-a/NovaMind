import React from 'react';
import { usePermissions } from '../../contexts/PermissionContext';
import { Alert, Box, Typography } from '@mui/material';
import { Security as SecurityIcon } from '@mui/icons-material';

const PermissionGuard = ({ 
  permission, 
  permissions = [], 
  any = false, 
  all = false,
  children, 
  fallback = null 
}) => {
  const { hasPermission, hasAnyPermission, hasAllPermissions } = usePermissions();

  const hasAccess = () => {
    if (permission) {
      return hasPermission(permission);
    }
    
    if (permissions.length > 0) {
      if (any) {
        return hasAnyPermission(permissions);
      }
      if (all) {
        return hasAllPermissions(permissions);
      }
      // Default to any permission
      return hasAnyPermission(permissions);
    }
    
    return true;
  };

  if (!hasAccess()) {
    if (fallback) {
      return fallback;
    }
    
    return (
      <Box
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        p={4}
        textAlign="center"
      >
        <SecurityIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h6" color="text.secondary" gutterBottom>
          Access Denied
        </Typography>
        <Typography variant="body2" color="text.secondary">
          You don't have the required permissions to access this resource.
        </Typography>
        {permission && (
          <Alert severity="info" sx={{ mt: 2, maxWidth: 400 }}>
            Required permission: <strong>{permission}</strong>
          </Alert>
        )}
        {permissions.length > 0 && (
          <Alert severity="info" sx={{ mt: 2, maxWidth: 400 }}>
            Required permissions: <strong>{permissions.join(', ')}</strong>
          </Alert>
        )}
      </Box>
    );
  }

  return children;
};

export default PermissionGuard; 