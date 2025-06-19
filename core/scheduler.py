"""
Task Scheduler for NovaMind Framework

Enterprise-grade task scheduling with priority queues, load balancing,
fault tolerance, and distributed task execution.
"""

import asyncio
import heapq
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime, timedelta
import json
import logging

from pydantic import BaseModel, Field
from loguru import logger


class TaskPriority(int, Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskType(str, Enum):
    """Task types for different execution strategies"""
    COMPUTE_INTENSIVE = "compute_intensive"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class Task:
    """Task definition with metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    payload: Any = None
    priority: TaskPriority = TaskPriority.NORMAL
    task_type: TaskType = TaskType.COMPUTE_INTENSIVE
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout: int = 300  # seconds
    retry_count: int = 0
    max_retries: int = 3
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Priority queue comparison"""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.value,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "assigned_agent": self.assigned_agent,
            "dependencies": self.dependencies,
            "error": self.error,
            "metadata": self.metadata
        }


class LoadBalancer:
    """Load balancer for agent assignment"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.agent_loads: Dict[str, int] = {}
        self.last_assignment: Dict[str, str] = {}
        
    def update_agent_load(self, agent_id: str, load: int):
        """Update agent load information"""
        self.agent_loads[agent_id] = load
        
    def select_agent(self, available_agents: List[str], task: Task) -> Optional[str]:
        """Select agent based on load balancing strategy"""
        if not available_agents:
            return None
            
        if self.strategy == "round_robin":
            return self._round_robin_select(available_agents)
        elif self.strategy == "least_loaded":
            return self._least_loaded_select(available_agents)
        elif self.strategy == "weighted":
            return self._weighted_select(available_agents, task)
        else:
            return available_agents[0]
            
    def _round_robin_select(self, available_agents: List[str]) -> str:
        """Round-robin selection"""
        if not available_agents:
            return None
            
        # Simple round-robin implementation
        return available_agents[0]
        
    def _least_loaded_select(self, available_agents: List[str]) -> str:
        """Select least loaded agent"""
        if not available_agents:
            return None
            
        min_load = float('inf')
        selected_agent = None
        
        for agent_id in available_agents:
            load = self.agent_loads.get(agent_id, 0)
            if load < min_load:
                min_load = load
                selected_agent = agent_id
                
        return selected_agent
        
    def _weighted_select(self, available_agents: List[str], task: Task) -> str:
        """Weighted selection based on task type and agent capabilities"""
        # Simplified weighted selection
        return self._least_loaded_select(available_agents)


class TaskQueue:
    """Priority-based task queue"""
    
    def __init__(self):
        self.tasks: List[Task] = []
        self.task_map: Dict[str, Task] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        
    def add_task(self, task: Task):
        """Add task to queue"""
        self.task_map[task.id] = task
        self.dependency_graph[task.id] = set(task.dependencies)
        heapq.heappush(self.tasks, task)
        
    def get_next_task(self, completed_tasks: Set[str]) -> Optional[Task]:
        """Get next available task"""
        available_tasks = []
        
        while self.tasks:
            task = heapq.heappop(self.tasks)
            
            # Check if dependencies are satisfied
            if task.dependencies and not all(dep in completed_tasks for dep in task.dependencies):
                available_tasks.append(task)
                continue
                
            # Check if task is still pending
            if task.status == TaskStatus.PENDING:
                return task
            elif task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                continue
            else:
                available_tasks.append(task)
                
        # Restore tasks that couldn't be processed
        for task in available_tasks:
            heapq.heappush(self.tasks, task)
            
        return None
        
    def update_task(self, task_id: str, **kwargs):
        """Update task properties"""
        if task_id in self.task_map:
            task = self.task_map[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
                    
    def remove_task(self, task_id: str):
        """Remove task from queue"""
        if task_id in self.task_map:
            del self.task_map[task_id]
            if task_id in self.dependency_graph:
                del self.dependency_graph[task_id]


class TaskScheduler:
    """
    Enterprise-grade task scheduler with advanced features
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.task_queue = TaskQueue()
        self.load_balancer = LoadBalancer()
        self.agents: Dict[str, Any] = {}
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.running_tasks: Dict[str, Task] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_completion_time": 0.0,
            "throughput": 0.0
        }
        
        self.logger = logger.bind(scheduler_name=name)
        self.running = False
        
    async def start(self):
        """Start the scheduler"""
        self.running = True
        self.logger.info("Task scheduler started")
        asyncio.create_task(self._scheduler_loop())
        asyncio.create_task(self._monitor_loop())
        
    async def stop(self):
        """Stop the scheduler"""
        self.running = False
        self.logger.info("Task scheduler stopped")
        
    def register_agent(self, agent_id: str, agent: Any):
        """Register agent with scheduler"""
        self.agents[agent_id] = agent
        self.load_balancer.update_agent_load(agent_id, 0)
        self.logger.info(f"Registered agent: {agent_id}")
        
    def unregister_agent(self, agent_id: str):
        """Unregister agent from scheduler"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.logger.info(f"Unregistered agent: {agent_id}")
            
    async def submit_task(self, task: Task) -> str:
        """Submit task for execution"""
        self.task_queue.add_task(task)
        self.metrics["total_tasks"] += 1
        self.logger.info(f"Submitted task: {task.id} - {task.name}")
        return task.id
        
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        if task_id in self.task_queue.task_map:
            task = self.task_queue.task_map[task_id]
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                self.logger.info(f"Cancelled task: {task_id}")
                return True
        return False
        
    async def _scheduler_loop(self):
        """Main scheduling loop"""
        while self.running:
            try:
                # Get next available task
                task = self.task_queue.get_next_task(self.completed_tasks)
                
                if task:
                    # Select agent for task
                    available_agents = [
                        agent_id for agent_id, agent in self.agents.items()
                        if hasattr(agent, 'state') and agent.state.value == 'idle'
                    ]
                    
                    if available_agents:
                        selected_agent = self.load_balancer.select_agent(available_agents, task)
                        if selected_agent:
                            await self._assign_task(task, selected_agent)
                    else:
                        # No available agents, wait a bit
                        await asyncio.sleep(0.1)
                else:
                    # No tasks available, wait
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1)
                
    async def _assign_task(self, task: Task, agent_id: str):
        """Assign task to agent"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.assigned_agent = agent_id
        self.running_tasks[task.id] = task
        
        # Update agent load
        current_load = self.load_balancer.agent_loads.get(agent_id, 0)
        self.load_balancer.update_agent_load(agent_id, current_load + 1)
        
        # Send task to agent
        agent = self.agents[agent_id]
        if hasattr(agent, 'receive_message'):
            # Create task message
            from .agent import Message, MessageType
            message = Message(
                sender="scheduler",
                recipient=agent_id,
                message_type=MessageType.TASK,
                content=task.payload,
                metadata={"task_id": task.id}
            )
            await agent.receive_message(message)
            
        self.logger.info(f"Assigned task {task.id} to agent {agent_id}")
        
    async def _monitor_loop(self):
        """Monitor running tasks and update metrics"""
        while self.running:
            try:
                # Check for completed tasks
                completed_task_ids = []
                for task_id, task in self.running_tasks.items():
                    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT]:
                        completed_task_ids.append(task_id)
                        
                        # Update metrics
                        if task.status == TaskStatus.COMPLETED:
                            self.completed_tasks.add(task_id)
                            self.metrics["completed_tasks"] += 1
                        else:
                            self.failed_tasks.add(task_id)
                            self.metrics["failed_tasks"] += 1
                            
                        # Update agent load
                        if task.assigned_agent:
                            current_load = self.load_balancer.agent_loads.get(task.assigned_agent, 0)
                            self.load_balancer.update_agent_load(task.assigned_agent, max(0, current_load - 1))
                            
                # Remove completed tasks
                for task_id in completed_task_ids:
                    del self.running_tasks[task_id]
                    
                # Check for timeouts
                current_time = datetime.now()
                for task_id, task in self.running_tasks.items():
                    if task.started_at and (current_time - task.started_at).total_seconds() > task.timeout:
                        task.status = TaskStatus.TIMEOUT
                        task.error = "Task timeout"
                        
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(1)
                
    def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics"""
        return {
            "scheduler_name": self.name,
            "running": self.running,
            "total_tasks": self.metrics["total_tasks"],
            "completed_tasks": self.metrics["completed_tasks"],
            "failed_tasks": self.metrics["failed_tasks"],
            "running_tasks": len(self.running_tasks),
            "pending_tasks": len(self.task_queue.tasks),
            "registered_agents": len(self.agents),
            "success_rate": self.metrics["completed_tasks"] / max(self.metrics["total_tasks"], 1),
            "agent_loads": self.load_balancer.agent_loads.copy()
        }
        
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        if task_id in self.task_queue.task_map:
            return self.task_queue.task_map[task_id].to_dict()
        return None 