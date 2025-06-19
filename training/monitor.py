"""
NovaMind训练监控系统 - 实时训练监控和可视化

提供实时训练监控功能：
- Web界面实时查看训练状态
- 训练指标可视化
- 系统资源监控
- 训练日志管理
- 模型性能分析
- 异常检测和告警
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
import queue

import psutil
import GPUtil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from loguru import logger
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from .trainer import TrainingManager, TrainingStatus
from ..core.monitor import SystemMonitor


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, port: int = 8080):
        """
        初始化训练监控器
        
        Args:
            port: Web服务器端口
        """
        self.port = port
        self.app = FastAPI(title="NovaMind Training Monitor", version="1.0.0")
        self.websocket_connections: List[WebSocket] = []
        self.metrics_buffer: Dict[str, List[Dict]] = {}
        self.system_monitor = SystemMonitor()
        
        # 设置路由
        self._setup_routes()
        
        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """获取监控仪表板"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/trainings")
        async def get_trainings():
            """获取所有训练状态"""
            return training_manager.list_trainings()
        
        @self.app.get("/api/training/{training_id}")
        async def get_training_status(training_id: str):
            """获取特定训练状态"""
            status = training_manager.get_training_status(training_id)
            if status is None:
                raise HTTPException(status_code=404, detail="训练未找到")
            return status
        
        @self.app.get("/api/metrics/{training_id}")
        async def get_training_metrics(training_id: str, limit: int = 100):
            """获取训练指标"""
            if training_id not in self.metrics_buffer:
                raise HTTPException(status_code=404, detail="训练指标未找到")
            
            metrics = self.metrics_buffer[training_id][-limit:]
            return {"metrics": metrics}
        
        @self.app.get("/api/system")
        async def get_system_info():
            """获取系统信息"""
            return self.system_monitor.get_system_info()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket连接"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # 发送实时数据
                    data = await self._get_realtime_data()
                    await websocket.send_text(json.dumps(data))
                    await asyncio.sleep(1)  # 每秒更新一次
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
    
    def _get_dashboard_html(self) -> str:
        """生成监控仪表板HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>NovaMind Training Monitor</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .training-list { grid-column: 1 / -1; }
                .metric-chart { height: 400px; }
                .status-badge { padding: 5px 10px; border-radius: 15px; color: white; font-size: 12px; }
                .status-running { background-color: #28a745; }
                .status-completed { background-color: #17a2b8; }
                .status-failed { background-color: #dc3545; }
                .status-paused { background-color: #ffc107; color: black; }
                .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
                .refresh-btn:hover { background: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 NovaMind Training Monitor</h1>
                    <p>实时训练监控和可视化系统</p>
                    <button class="refresh-btn" onclick="refreshData()">刷新数据</button>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <h3>📊 训练指标</h3>
                        <div id="lossChart" class="metric-chart"></div>
                    </div>
                    
                    <div class="card">
                        <h3>⚡ 系统资源</h3>
                        <div id="systemChart" class="metric-chart"></div>
                    </div>
                    
                    <div class="card training-list">
                        <h3>🎯 训练任务</h3>
                        <div id="trainingList"></div>
                    </div>
                </div>
            </div>
            
            <script>
                let ws = new WebSocket('ws://localhost:8080/ws');
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                function updateDashboard(data) {
                    updateTrainingList(data.trainings);
                    updateLossChart(data.metrics);
                    updateSystemChart(data.system);
                }
                
                function updateTrainingList(trainings) {
                    const container = document.getElementById('trainingList');
                    container.innerHTML = '';
                    
                    trainings.forEach(training => {
                        const div = document.createElement('div');
                        div.style.cssText = 'border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;';
                        
                        const statusClass = 'status-' + training.status.status;
                        div.innerHTML = `
                            <h4>${training.id}</h4>
                            <p><strong>状态:</strong> <span class="status-badge ${statusClass}">${training.status.status}</span></p>
                            <p><strong>当前轮数:</strong> ${training.status.current_epoch}</p>
                            <p><strong>当前步数:</strong> ${training.status.current_step}</p>
                            <p><strong>最佳损失:</strong> ${training.status.best_loss.toFixed(4)}</p>
                        `;
                        
                        container.appendChild(div);
                    });
                }
                
                function updateLossChart(metrics) {
                    if (!metrics || metrics.length === 0) return;
                    
                    const traces = [];
                    const trainingIds = [...new Set(metrics.map(m => m.training_id))];
                    
                    trainingIds.forEach(trainingId => {
                        const trainingMetrics = metrics.filter(m => m.training_id === trainingId);
                        traces.push({
                            x: trainingMetrics.map(m => m.step),
                            y: trainingMetrics.map(m => m.loss),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: trainingId,
                            line: { width: 2 }
                        });
                    });
                    
                    const layout = {
                        title: '训练损失曲线',
                        xaxis: { title: '步数' },
                        yaxis: { title: '损失' },
                        hovermode: 'closest'
                    };
                    
                    Plotly.newPlot('lossChart', traces, layout);
                }
                
                function updateSystemChart(system) {
                    if (!system) return;
                    
                    const data = [
                        {
                            values: [system.cpu_percent, 100 - system.cpu_percent],
                            labels: ['CPU使用率', '空闲'],
                            type: 'pie',
                            name: 'CPU',
                            domain: { row: 0, column: 0 }
                        },
                        {
                            values: [system.memory_percent, 100 - system.memory_percent],
                            labels: ['内存使用率', '空闲'],
                            type: 'pie',
                            name: '内存',
                            domain: { row: 0, column: 1 }
                        }
                    ];
                    
                    const layout = {
                        title: '系统资源使用情况',
                        grid: { rows: 1, columns: 2 }
                    };
                    
                    Plotly.newPlot('systemChart', data, layout);
                }
                
                function refreshData() {
                    fetch('/api/trainings')
                        .then(response => response.json())
                        .then(data => {
                            updateTrainingList(data);
                        });
                }
                
                // 初始加载
                refreshData();
            </script>
        </body>
        </html>
        """
    
    async def _get_realtime_data(self) -> Dict[str, Any]:
        """获取实时数据"""
        # 获取训练状态
        trainings = training_manager.list_trainings()
        
        # 获取系统信息
        system_info = self.system_monitor.get_system_info()
        
        # 获取指标数据
        metrics = []
        for training_id in self.metrics_buffer:
            if self.metrics_buffer[training_id]:
                latest_metrics = self.metrics_buffer[training_id][-10:]  # 最近10个指标
                for metric in latest_metrics:
                    metric['training_id'] = training_id
                    metrics.append(metric)
        
        return {
            'trainings': trainings,
            'system': system_info,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def _monitoring_loop(self):
        """监控循环"""
        while True:
            try:
                # 更新系统监控
                self.system_monitor.update()
                
                # 广播数据到WebSocket连接
                asyncio.run(self._broadcast_data())
                
                time.sleep(1)  # 每秒更新一次
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(5)
    
    async def _broadcast_data(self):
        """广播数据到WebSocket连接"""
        if not self.websocket_connections:
            return
        
        data = await self._get_realtime_data()
        message = json.dumps(data)
        
        # 发送到所有连接的客户端
        for websocket in self.websocket_connections[:]:  # 复制列表避免修改
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"WebSocket发送错误: {e}")
                self.websocket_connections.remove(websocket)
    
    def add_metrics(self, training_id: str, metrics: Dict[str, Any]):
        """
        添加训练指标
        
        Args:
            training_id: 训练ID
            metrics: 指标数据
        """
        if training_id not in self.metrics_buffer:
            self.metrics_buffer[training_id] = []
        
        # 添加时间戳
        metrics['timestamp'] = datetime.now().isoformat()
        
        self.metrics_buffer[training_id].append(metrics)
        
        # 限制缓冲区大小
        if len(self.metrics_buffer[training_id]) > 1000:
            self.metrics_buffer[training_id] = self.metrics_buffer[training_id][-500:]
    
    def get_training_metrics(self, training_id: str, limit: int = 100) -> List[Dict]:
        """
        获取训练指标
        
        Args:
            training_id: 训练ID
            limit: 返回的指标数量限制
            
        Returns:
            List[Dict]: 指标列表
        """
        if training_id not in self.metrics_buffer:
            return []
        
        return self.metrics_buffer[training_id][-limit:]
    
    def generate_metrics_report(self, training_id: str) -> Dict[str, Any]:
        """
        生成指标报告
        
        Args:
            training_id: 训练ID
            
        Returns:
            Dict[str, Any]: 报告数据
        """
        if training_id not in self.metrics_buffer:
            return {}
        
        metrics = self.metrics_buffer[training_id]
        if not metrics:
            return {}
        
        # 转换为DataFrame进行分析
        df = pd.DataFrame(metrics)
        
        report = {
            'training_id': training_id,
            'total_steps': len(metrics),
            'start_time': metrics[0]['timestamp'] if metrics else None,
            'end_time': metrics[-1]['timestamp'] if metrics else None,
            'final_loss': metrics[-1]['loss'] if metrics else None,
            'best_loss': min(m['loss'] for m in metrics) if metrics else None,
            'avg_loss': df['loss'].mean() if 'loss' in df.columns else None,
            'loss_std': df['loss'].std() if 'loss' in df.columns else None,
            'final_accuracy': metrics[-1].get('accuracy', 0) if metrics else None,
            'best_accuracy': max(m.get('accuracy', 0) for m in metrics) if metrics else None,
            'avg_learning_rate': df['learning_rate'].mean() if 'learning_rate' in df.columns else None,
            'total_training_time': None  # 需要计算
        }
        
        # 计算训练时间
        if report['start_time'] and report['end_time']:
            start = datetime.fromisoformat(report['start_time'])
            end = datetime.fromisoformat(report['end_time'])
            report['total_training_time'] = str(end - start)
        
        return report
    
    def create_visualization(self, training_id: str) -> Dict[str, Any]:
        """
        创建可视化图表
        
        Args:
            training_id: 训练ID
            
        Returns:
            Dict[str, Any]: 图表数据
        """
        if training_id not in self.metrics_buffer:
            return {}
        
        metrics = self.metrics_buffer[training_id]
        if not metrics:
            return {}
        
        df = pd.DataFrame(metrics)
        
        # 创建损失曲线
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(
            x=df['step'],
            y=df['loss'],
            mode='lines+markers',
            name='训练损失',
            line=dict(color='red', width=2)
        ))
        loss_fig.update_layout(
            title='训练损失曲线',
            xaxis_title='步数',
            yaxis_title='损失',
            hovermode='closest'
        )
        
        # 创建准确率曲线
        accuracy_fig = go.Figure()
        if 'accuracy' in df.columns:
            accuracy_fig.add_trace(go.Scatter(
                x=df['step'],
                y=df['accuracy'],
                mode='lines+markers',
                name='准确率',
                line=dict(color='blue', width=2)
            ))
            accuracy_fig.update_layout(
                title='准确率曲线',
                xaxis_title='步数',
                yaxis_title='准确率',
                hovermode='closest'
            )
        
        # 创建学习率曲线
        lr_fig = go.Figure()
        if 'learning_rate' in df.columns:
            lr_fig.add_trace(go.Scatter(
                x=df['step'],
                y=df['learning_rate'],
                mode='lines',
                name='学习率',
                line=dict(color='green', width=2)
            ))
            lr_fig.update_layout(
                title='学习率变化',
                xaxis_title='步数',
                yaxis_title='学习率',
                hovermode='closest'
            )
        
        return {
            'loss_chart': loss_fig.to_json(),
            'accuracy_chart': accuracy_fig.to_json() if 'accuracy' in df.columns else None,
            'lr_chart': lr_fig.to_json() if 'learning_rate' in df.columns else None
        }
    
    def start(self):
        """启动监控服务器"""
        logger.info(f"启动训练监控服务器，端口: {self.port}")
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)
    
    def start_background(self):
        """在后台启动监控服务器"""
        def run_server():
            uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="error")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        logger.info(f"训练监控服务器已在后台启动，端口: {self.port}")


# 全局监控器实例
training_monitor = TrainingMonitor()


class MetricsCallback:
    """指标回调函数"""
    
    def __init__(self, training_id: str):
        """
        初始化指标回调
        
        Args:
            training_id: 训练ID
        """
        self.training_id = training_id
    
    def __call__(self, trainer, metrics: Dict[str, float]):
        """
        回调函数
        
        Args:
            trainer: 训练器实例
            metrics: 训练指标
        """
        # 添加训练ID到指标中
        metrics['training_id'] = self.training_id
        
        # 发送到监控器
        training_monitor.add_metrics(self.training_id, metrics) 