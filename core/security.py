"""
Security Manager for NovaMind Framework

Enterprise-grade security with access control, authentication,
authorization, audit logging, and encryption.
"""

import hashlib
import hmac
import jwt
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import json
import logging

from pydantic import BaseModel, Field
from loguru import logger


class Permission(str, Enum):
    """System permissions"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    CREATE = "create"
    DELETE = "delete"
    UPDATE = "update"


class ResourceType(str, Enum):
    """Resource types for access control"""
    AGENT = "agent"
    MODEL = "model"
    TOOL = "tool"
    MEMORY = "memory"
    CONFIG = "config"
    SYSTEM = "system"
    API = "api"


class SecurityLevel(str, Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class User:
    """User definition"""
    id: str
    username: str
    email: str
    password_hash: str
    roles: List[str] = field(default_factory=list)
    permissions: Set[Permission] = field(default_factory=set)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "roles": self.roles,
            "permissions": [p.value for p in self.permissions],
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "security_level": self.security_level.value
        }


@dataclass
class Role:
    """Role definition"""
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    resources: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "permissions": [p.value for p in self.permissions],
            "resources": self.resources,
            "security_level": self.security_level.value
        }


@dataclass
class AuditLog:
    """Audit log entry"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    action: str = ""
    resource: str = ""
    resource_type: ResourceType = ResourceType.SYSTEM
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: str = ""
    user_agent: str = ""
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "resource_type": self.resource_type.value,
            "timestamp": self.timestamp.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success,
            "details": self.details
        }


class AccessControl:
    """Access control system"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.resource_permissions: Dict[str, Dict[str, Set[Permission]]] = {}
        self.audit_logs: List[AuditLog] = []
        
        # Initialize default roles
        self._setup_default_roles()
        
    def _setup_default_roles(self):
        """Setup default system roles"""
        # Admin role
        admin_role = Role(
            name="admin",
            description="System administrator with full access",
            permissions={Permission.READ, Permission.WRITE, Permission.EXECUTE, 
                        Permission.ADMIN, Permission.CREATE, Permission.DELETE, Permission.UPDATE},
            security_level=SecurityLevel.CRITICAL
        )
        self.roles["admin"] = admin_role
        
        # User role
        user_role = Role(
            name="user",
            description="Standard user with basic access",
            permissions={Permission.READ, Permission.EXECUTE},
            security_level=SecurityLevel.MEDIUM
        )
        self.roles["user"] = user_role
        
        # Guest role
        guest_role = Role(
            name="guest",
            description="Guest user with limited access",
            permissions={Permission.READ},
            security_level=SecurityLevel.LOW
        )
        self.roles["guest"] = guest_role
        
    def add_user(self, user: User):
        """Add user to the system"""
        self.users[user.id] = user
        logger.info(f"Added user: {user.username}")
        
    def remove_user(self, user_id: str):
        """Remove user from the system"""
        if user_id in self.users:
            del self.users[user_id]
            logger.info(f"Removed user: {user_id}")
            
    def add_role(self, role: Role):
        """Add role to the system"""
        self.roles[role.name] = role
        logger.info(f"Added role: {role.name}")
        
    def assign_role_to_user(self, user_id: str, role_name: str):
        """Assign role to user"""
        if user_id in self.users and role_name in self.roles:
            self.users[user_id].roles.append(role_name)
            # Add role permissions to user
            role = self.roles[role_name]
            self.users[user_id].permissions.update(role.permissions)
            logger.info(f"Assigned role {role_name} to user {user_id}")
            
    def check_permission(self, user_id: str, resource: str, permission: Permission) -> bool:
        """Check if user has permission for resource"""
        if user_id not in self.users:
            return False
            
        user = self.users[user_id]
        if not user.is_active:
            return False
            
        # Check direct permissions
        if permission in user.permissions:
            return True
            
        # Check role-based permissions
        for role_name in user.roles:
            if role_name in self.roles:
                role = self.roles[role_name]
                if permission in role.permissions:
                    return True
                    
        return False
        
    def log_audit_event(self, audit_log: AuditLog):
        """Log audit event"""
        self.audit_logs.append(audit_log)
        # Keep only last 10000 audit logs
        if len(self.audit_logs) > 10000:
            self.audit_logs = self.audit_logs[-10000:]
            
    def get_audit_logs(self, user_id: Optional[str] = None, 
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> List[AuditLog]:
        """Get audit logs with optional filtering"""
        logs = self.audit_logs
        
        if user_id:
            logs = [log for log in logs if log.user_id == user_id]
            
        if start_time:
            logs = [log for log in logs if log.timestamp >= start_time]
            
        if end_time:
            logs = [log for log in logs if log.timestamp <= end_time]
            
        return logs


class SecurityManager:
    """
    Enterprise-grade security manager
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.access_control = AccessControl()
        self.jwt_secret = secrets.token_hex(32)
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_hours = 24
        
        # Security policies
        self.password_policy = {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digits": True,
            "require_special": True
        }
        
        self.rate_limiting = {
            "max_requests_per_minute": 100,
            "max_failed_attempts": 5,
            "lockout_duration_minutes": 15
        }
        
        # Rate limiting tracking
        self.request_counts: Dict[str, List[datetime]] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.locked_accounts: Set[str] = set()
        
        self.logger = logger.bind(security_manager="default")
        
    def hash_password(self, password: str) -> str:
        """Hash password using secure algorithm"""
        salt = secrets.token_hex(16)
        hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${hash_obj.hex()}"
        
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_hex = password_hash.split('$')
            hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hmac.compare_digest(hash_obj.hex(), hash_hex)
        except Exception:
            return False
            
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password against policy"""
        errors = []
        
        if len(password) < self.password_policy["min_length"]:
            errors.append(f"Password must be at least {self.password_policy['min_length']} characters")
            
        if self.password_policy["require_uppercase"] and not any(c.isupper() for c in password):
            errors.append("Password must contain uppercase letter")
            
        if self.password_policy["require_lowercase"] and not any(c.islower() for c in password):
            errors.append("Password must contain lowercase letter")
            
        if self.password_policy["require_digits"] and not any(c.isdigit() for c in password):
            errors.append("Password must contain digit")
            
        if self.password_policy["require_special"] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain special character")
            
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
        
    def create_user(self, username: str, email: str, password: str, 
                   roles: List[str] = None) -> Optional[User]:
        """Create new user"""
        # Validate password
        validation = self.validate_password(password)
        if not validation["valid"]:
            self.logger.warning(f"Password validation failed: {validation['errors']}")
            return None
            
        # Check if user already exists
        for user in self.access_control.users.values():
            if user.username == username or user.email == email:
                self.logger.warning(f"User already exists: {username}")
                return None
                
        # Create user
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            password_hash=self.hash_password(password),
            roles=roles or ["user"]
        )
        
        # Assign roles
        for role_name in user.roles:
            self.access_control.assign_role_to_user(user.id, role_name)
            
        self.access_control.add_user(user)
        
        # Log audit event
        audit_log = AuditLog(
            user_id="system",
            action="user_created",
            resource=user.id,
            resource_type=ResourceType.SYSTEM,
            details={"username": username, "email": email}
        )
        self.access_control.log_audit_event(audit_log)
        
        return user
        
    def authenticate_user(self, username: str, password: str, ip_address: str = "") -> Optional[str]:
        """Authenticate user and return JWT token"""
        # Check if account is locked
        if username in self.locked_accounts:
            self.logger.warning(f"Login attempt for locked account: {username}")
            return None
            
        # Find user
        user = None
        for u in self.access_control.users.values():
            if u.username == username:
                user = u
                break
                
        if not user or not user.is_active:
            self._record_failed_attempt(username, ip_address)
            return None
            
        # Verify password
        if not self.verify_password(password, user.password_hash):
            self._record_failed_attempt(username, ip_address)
            return None
            
        # Update last login
        user.last_login = datetime.now()
        
        # Generate JWT token
        token = self.generate_jwt_token(user.id)
        
        # Log successful login
        audit_log = AuditLog(
            user_id=user.id,
            action="login_success",
            resource="auth",
            resource_type=ResourceType.SYSTEM,
            ip_address=ip_address,
            success=True
        )
        self.access_control.log_audit_event(audit_log)
        
        return token
        
    def generate_jwt_token(self, user_id: str) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=self.jwt_expiry_hours),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
    def verify_jwt_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return user ID"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload.get("user_id")
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid JWT token")
            return None
            
    def check_access(self, token: str, resource: str, permission: Permission) -> bool:
        """Check if user has access to resource"""
        user_id = self.verify_jwt_token(token)
        if not user_id:
            return False
            
        return self.access_control.check_permission(user_id, resource, permission)
        
    def _record_failed_attempt(self, username: str, ip_address: str):
        """Record failed login attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
            
        self.failed_attempts[username].append(datetime.now())
        
        # Remove old attempts
        cutoff_time = datetime.now() - timedelta(minutes=self.rate_limiting["lockout_duration_minutes"])
        self.failed_attempts[username] = [
            attempt for attempt in self.failed_attempts[username]
            if attempt > cutoff_time
        ]
        
        # Check if account should be locked
        if len(self.failed_attempts[username]) >= self.rate_limiting["max_failed_attempts"]:
            self.locked_accounts.add(username)
            self.logger.warning(f"Account locked due to failed attempts: {username}")
            
        # Log failed attempt
        audit_log = AuditLog(
            user_id=username,
            action="login_failed",
            resource="auth",
            resource_type=ResourceType.SYSTEM,
            ip_address=ip_address,
            success=False
        )
        self.access_control.log_audit_event(audit_log)
        
    def unlock_account(self, username: str):
        """Unlock user account"""
        if username in self.locked_accounts:
            self.locked_accounts.remove(username)
            if username in self.failed_attempts:
                del self.failed_attempts[username]
            self.logger.info(f"Account unlocked: {username}")
            
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security system summary"""
        return {
            "total_users": len(self.access_control.users),
            "active_users": len([u for u in self.access_control.users.values() if u.is_active]),
            "total_roles": len(self.access_control.roles),
            "locked_accounts": len(self.locked_accounts),
            "audit_logs_count": len(self.access_control.audit_logs),
            "security_policies": {
                "password_policy": self.password_policy,
                "rate_limiting": self.rate_limiting
            }
        }
        
    def export_audit_logs(self, format: str = "json") -> str:
        """Export audit logs"""
        logs = [log.to_dict() for log in self.access_control.audit_logs]
        if format == "json":
            return json.dumps(logs, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}") 