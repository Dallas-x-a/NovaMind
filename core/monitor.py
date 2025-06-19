"""
System Monitor for NovaMind Framework

Enterprise-grade monitoring with performance metrics, alerting,
visualization, and health checks.
"""

import asyncio
import time
import psutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Set
from enum import Enum
import json
import statistics

from pydantic import BaseModel, Field
from loguru import logger


class MetricType(str, Enum):
    """Metric types for monitoring"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Performance metric definition"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "description": self.description
        }


@dataclass
class Alert:
    """System alert definition"""
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved
        }


class HealthStatus(str, Enum):
    """System health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class PerformanceMetrics:
    """Performance metrics collector"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Metric]] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_handlers: List[Callable] = []
        self.health_checks: Dict[str, Callable] = {}
        
    def record_metric(self, metric: Metric):
        """Record a new metric"""
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
            
        self.metrics[metric.name].append(metric)
        
        # Keep only last 1000 metrics per name
        if len(self.metrics[metric.name]) > 1000:
            self.metrics[metric.name] = self.metrics[metric.name][-1000:]
            
        # Check alert rules
        self._check_alerts(metric)
        
    def get_metric(self, name: str, window_minutes: int = 60) -> List[Metric]:
        """Get metrics for a specific name within time window"""
        if name not in self.metrics:
            return []
            
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        return [
            metric for metric in self.metrics[name]
            if metric.timestamp >= cutoff_time
        ]
        
    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """Get the latest metric for a name"""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1]
        return None
        
    def get_metric_summary(self, name: str, window_minutes: int = 60) -> Dict[str, float]:
        """Get statistical summary of metrics"""
        metrics = self.get_metric(name, window_minutes)
        if not metrics:
            return {}
            
        values = [m.value for m in metrics]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0
        }
        
    def add_alert_rule(self, metric_name: str, threshold: float, level: AlertLevel, message: str):
        """Add alert rule for a metric"""
        self.alert_rules[metric_name] = {
            "threshold": threshold,
            "level": level,
            "message": message
        }
        
    def add_alert_handler(self, handler: Callable):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
        
    def add_health_check(self, name: str, check_func: Callable):
        """Add health check function"""
        self.health_checks[name] = check_func
        
    def _check_alerts(self, metric: Metric):
        """Check if metric triggers any alerts"""
        if metric.name in self.alert_rules:
            rule = self.alert_rules[metric.name]
            
            # Simple threshold check (can be extended for more complex rules)
            if metric.value > rule["threshold"]:
                alert = Alert(
                    id=f"{metric.name}_{int(time.time())}",
                    level=rule["level"],
                    message=rule["message"],
                    metric_name=metric.name,
                    threshold=rule["threshold"],
                    current_value=metric.value
                )
                
                # Trigger alert handlers
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Alert handler error: {e}")
                        
    def get_all_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all metrics as dictionaries"""
        return {
            name: [metric.to_dict() for metric in metrics]
            for name, metrics in self.metrics.items()
        }


class SystemMonitor:
    """
    Enterprise-grade system monitor with comprehensive metrics collection
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.metrics = PerformanceMetrics()
        self.running = False
        self.monitoring_interval = 5  # seconds
        self.logger = logger.bind(monitor_name=name)
        
        # System metrics
        self._setup_system_metrics()
        self._setup_alert_rules()
        
    def _setup_system_metrics(self):
        """Setup system-level metrics collection"""
        # CPU usage
        self.metrics.add_health_check("cpu_usage", self._check_cpu_usage)
        
        # Memory usage
        self.metrics.add_health_check("memory_usage", self._check_memory_usage)
        
        # Disk usage
        self.metrics.add_health_check("disk_usage", self._check_disk_usage)
        
        # Network I/O
        self.metrics.add_health_check("network_io", self._check_network_io)
        
    def _setup_alert_rules(self):
        """Setup default alert rules"""
        # CPU usage alerts
        self.metrics.add_alert_rule(
            "cpu_usage_percent",
            80.0,
            AlertLevel.WARNING,
            "CPU usage is high"
        )
        self.metrics.add_alert_rule(
            "cpu_usage_percent",
            95.0,
            AlertLevel.CRITICAL,
            "CPU usage is critical"
        )
        
        # Memory usage alerts
        self.metrics.add_alert_rule(
            "memory_usage_percent",
            85.0,
            AlertLevel.WARNING,
            "Memory usage is high"
        )
        self.metrics.add_alert_rule(
            "memory_usage_percent",
            95.0,
            AlertLevel.CRITICAL,
            "Memory usage is critical"
        )
        
        # Disk usage alerts
        self.metrics.add_alert_rule(
            "disk_usage_percent",
            90.0,
            AlertLevel.WARNING,
            "Disk usage is high"
        )
        
    async def start(self):
        """Start the system monitor"""
        self.running = True
        self.logger.info("System monitor started")
        asyncio.create_task(self._monitoring_loop())
        
    async def stop(self):
        """Stop the system monitor"""
        self.running = False
        self.logger.info("System monitor stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Run health checks
                await self._run_health_checks()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
                
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.record_metric(Metric(
            name="cpu_usage_percent",
            value=cpu_percent,
            metric_type=MetricType.GAUGE,
            description="CPU usage percentage"
        ))
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics.record_metric(Metric(
            name="memory_usage_percent",
            value=memory.percent,
            metric_type=MetricType.GAUGE,
            description="Memory usage percentage"
        ))
        
        self.metrics.record_metric(Metric(
            name="memory_available_gb",
            value=memory.available / (1024**3),
            metric_type=MetricType.GAUGE,
            description="Available memory in GB"
        ))
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.metrics.record_metric(Metric(
            name="disk_usage_percent",
            value=(disk.used / disk.total) * 100,
            metric_type=MetricType.GAUGE,
            description="Disk usage percentage"
        ))
        
        # Network I/O
        network = psutil.net_io_counters()
        self.metrics.record_metric(Metric(
            name="network_bytes_sent",
            value=network.bytes_sent,
            metric_type=MetricType.COUNTER,
            description="Network bytes sent"
        ))
        
        self.metrics.record_metric(Metric(
            name="network_bytes_recv",
            value=network.bytes_recv,
            metric_type=MetricType.COUNTER,
            description="Network bytes received"
        ))
        
    async def _run_health_checks(self):
        """Run health check functions"""
        for name, check_func in self.health_checks.items():
            try:
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                self.metrics.record_metric(Metric(
                    name=f"health_check_{name}",
                    value=1.0 if result else 0.0,
                    metric_type=MetricType.GAUGE,
                    description=f"Health check result for {name}"
                ))
            except Exception as e:
                self.logger.error(f"Health check {name} failed: {e}")
                self.metrics.record_metric(Metric(
                    name=f"health_check_{name}",
                    value=0.0,
                    metric_type=MetricType.GAUGE,
                    description=f"Health check result for {name}"
                ))
                
    def _check_cpu_usage(self) -> bool:
        """Check if CPU usage is healthy"""
        latest = self.metrics.get_latest_metric("cpu_usage_percent")
        return latest is None or latest.value < 90.0
        
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is healthy"""
        latest = self.metrics.get_latest_metric("memory_usage_percent")
        return latest is None or latest.value < 95.0
        
    def _check_disk_usage(self) -> bool:
        """Check if disk usage is healthy"""
        latest = self.metrics.get_latest_metric("disk_usage_percent")
        return latest is None or latest.value < 95.0
        
    def _check_network_io(self) -> bool:
        """Check if network I/O is healthy"""
        # Simple network health check
        return True
        
    def get_system_health(self) -> HealthStatus:
        """Get overall system health status"""
        health_checks = []
        
        for name in self.health_checks.keys():
            latest = self.metrics.get_latest_metric(f"health_check_{name}")
            if latest:
                health_checks.append(latest.value > 0.5)
                
        if not health_checks:
            return HealthStatus.UNKNOWN
            
        healthy_count = sum(health_checks)
        total_count = len(health_checks)
        
        if healthy_count == total_count:
            return HealthStatus.HEALTHY
        elif healthy_count > total_count * 0.5:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY
            
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system monitoring summary"""
        return {
            "monitor_name": self.name,
            "running": self.running,
            "health_status": self.get_system_health().value,
            "metrics_count": len(self.metrics.metrics),
            "alert_rules_count": len(self.metrics.alert_rules),
            "health_checks_count": len(self.metrics.health_checks),
            "current_metrics": {
                "cpu_usage": self.metrics.get_latest_metric("cpu_usage_percent"),
                "memory_usage": self.metrics.get_latest_metric("memory_usage_percent"),
                "disk_usage": self.metrics.get_latest_metric("disk_usage_percent")
            }
        }
        
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        if format == "json":
            return json.dumps(self.metrics.get_all_metrics(), indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


class AgentMonitor:
    """Monitor for agent-specific metrics"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.metrics = PerformanceMetrics()
        self.logger = logger.bind(agent_id=agent_id)
        
    def record_agent_metric(self, metric_name: str, value: float, description: str = ""):
        """Record agent-specific metric"""
        self.metrics.record_metric(Metric(
            name=metric_name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels={"agent_id": self.agent_id},
            description=description
        ))
        
    def record_task_metric(self, task_id: str, duration: float, success: bool):
        """Record task execution metric"""
        self.metrics.record_metric(Metric(
            name="task_duration_seconds",
            value=duration,
            metric_type=MetricType.HISTOGRAM,
            labels={"agent_id": self.agent_id, "task_id": task_id, "success": str(success).lower()},
            description="Task execution duration"
        ))
        
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get agent monitoring summary"""
        return {
            "agent_id": self.agent_id,
            "metrics_count": len(self.metrics.metrics),
            "task_duration_summary": self.metrics.get_metric_summary("task_duration_seconds"),
            "recent_metrics": {
                name: self.metrics.get_latest_metric(name)
                for name in self.metrics.metrics.keys()
            }
        } 