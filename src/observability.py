#!/usr/bin/env python3
"""
Comprehensive Logging and Observability System for MLOps
Advanced logging, tracing, and observability with OpenTelemetry integration
"""

import os
import sys
import json
import logging
import traceback
import inspect
import functools
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import threading
from collections import defaultdict, deque
import socket
import psutil

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.logging import LoggingInstrumentor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: str
    process_id: int
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    model_name: Optional[str] = None
    environment: str = "unknown"
    extra_fields: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricPoint:
    """Metric data point"""
    timestamp: datetime
    name: str
    value: float
    unit: str
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""

@dataclass
class TraceSpan:
    """Distributed tracing span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    status: str  # success, error, timeout
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

class StructuredLogger:
    """Enhanced structured logger with context management"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
        
        # Setup custom formatter
        formatter = StructuredFormatter()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if configured
        if self.config.get('file_logging', {}).get('enabled', False):
            file_path = self.config['file_logging'].get('path', 'logs/mlops.log')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        # Context storage
        self._context = threading.local()
        
    def set_context(self, **kwargs):
        """Set logging context for current thread"""
        if not hasattr(self._context, 'data'):
            self._context.data = {}
        self._context.data.update(kwargs)
        
    def get_context(self) -> Dict[str, Any]:
        """Get current logging context"""
        if hasattr(self._context, 'data'):
            return self._context.data.copy()
        return {}
        
    def clear_context(self):
        """Clear logging context"""
        if hasattr(self._context, 'data'):
            self._context.data.clear()
            
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with current context"""
        # Get caller information
        frame = inspect.currentframe().f_back.f_back
        
        # Create structured log entry
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=level.upper(),
            logger_name=self.name,
            message=message,
            module=frame.f_globals.get('__name__', 'unknown'),
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            thread_id=str(threading.current_thread().ident),
            process_id=os.getpid(),
            environment=os.getenv('ENVIRONMENT', 'development'),
            extra_fields=kwargs
        )
        
        # Add context
        context = self.get_context()
        for key, value in context.items():
            if hasattr(log_entry, key):
                setattr(log_entry, key, value)
            else:
                log_entry.extra_fields[key] = value
                
        # Add tracing information if available
        current_span = trace.get_current_span()
        if current_span:
            span_context = current_span.get_span_context()
            log_entry.trace_id = format(span_context.trace_id, '032x')
            log_entry.span_id = format(span_context.span_id, '016x')
            
        # Convert to dict for logging
        log_dict = asdict(log_entry)
        log_dict['timestamp'] = log_entry.timestamp.isoformat()
        
        # Log using standard logger
        getattr(self.logger, level.lower())(json.dumps(log_dict))
        
    def debug(self, message: str, **kwargs):
        self._log_with_context('debug', message, **kwargs)
        
    def info(self, message: str, **kwargs):
        self._log_with_context('info', message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        self._log_with_context('warning', message, **kwargs)
        
    def error(self, message: str, **kwargs):
        self._log_with_context('error', message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        self._log_with_context('critical', message, **kwargs)

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def format(self, record):
        # If the message is already JSON, return as-is
        try:
            json.loads(record.getMessage())
            return record.getMessage()
        except (json.JSONDecodeError, ValueError):
            # Create structured log entry for non-JSON messages
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger_name': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line_number': record.lineno,
                'thread_id': str(threading.current_thread().ident),
                'process_id': os.getpid()
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = {
                    'type': record.exc_info[0].__name__,
                    'message': str(record.exc_info[1]),
                    'traceback': traceback.format_exception(*record.exc_info)
                }
                
            return json.dumps(log_entry)

class MetricsCollector:
    """Advanced metrics collection and aggregation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_buffer = deque(maxlen=10000)
        self.aggregated_metrics = defaultdict(list)
        
        # Initialize OpenTelemetry metrics
        self._setup_otel_metrics()
        
        # Built-in metrics
        self.counters = defaultdict(int)
        self.histograms = defaultdict(list)
        self.gauges = defaultdict(float)
        
        # System metrics collection
        self.collect_system_metrics = self.config.get('system_metrics', True)
        if self.collect_system_metrics:
            self._start_system_metrics_collection()
            
    def _setup_otel_metrics(self):
        """Setup OpenTelemetry metrics"""
        resource = Resource.create({
            "service.name": "mlops-pipeline",
            "service.version": "1.0.0"
        })
        
        # Setup Prometheus exporter if configured
        if self.config.get('prometheus', {}).get('enabled', False):
            prometheus_reader = PrometheusMetricReader()
            metrics.set_meter_provider(MeterProvider(
                resource=resource,
                metric_readers=[prometheus_reader]
            ))
        else:
            metrics.set_meter_provider(MeterProvider(resource=resource))
            
        self.meter = metrics.get_meter(__name__)
        
        # Create common metrics
        self.request_counter = self.meter.create_counter(
            "mlops_requests_total",
            description="Total number of requests"
        )
        
        self.latency_histogram = self.meter.create_histogram(
            "mlops_request_duration_seconds",
            description="Request duration in seconds"
        )
        
        self.model_predictions = self.meter.create_counter(
            "mlops_model_predictions_total",
            description="Total number of model predictions"
        )
        
        self.model_accuracy = self.meter.create_up_down_counter(
            "mlops_model_accuracy",
            description="Current model accuracy"
        )
        
    def _start_system_metrics_collection(self):
        """Start background system metrics collection"""
        def collect_system_metrics():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.record_gauge("system.cpu.usage_percent", cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.record_gauge("system.memory.usage_percent", memory.percent)
                    self.record_gauge("system.memory.available_bytes", memory.available)
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    self.record_gauge("system.disk.usage_percent", 
                                    (disk.used / disk.total) * 100)
                    
                    # Network I/O
                    network = psutil.net_io_counters()
                    self.record_counter("system.network.bytes_sent", network.bytes_sent)
                    self.record_counter("system.network.bytes_recv", network.bytes_recv)
                    
                    time.sleep(30)  # Collect every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(60)
                    
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
        
    def record_counter(self, name: str, value: float = 1, labels: Dict[str, str] = None):
        """Record counter metric"""
        labels = labels or {}
        
        # Record in internal storage
        self.counters[name] += value
        
        # Record in OpenTelemetry
        if hasattr(self, 'request_counter') and 'request' in name:
            self.request_counter.add(value, labels)
        elif hasattr(self, 'model_predictions') and 'prediction' in name:
            self.model_predictions.add(value, labels)
            
        # Add to buffer
        metric_point = MetricPoint(
            timestamp=datetime.now(),
            name=name,
            value=value,
            unit="count",
            labels=labels,
            description=f"Counter metric: {name}"
        )
        self.metrics_buffer.append(metric_point)
        
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record histogram metric"""
        labels = labels or {}
        
        # Record in internal storage
        self.histograms[name].append(value)
        
        # Record in OpenTelemetry
        if hasattr(self, 'latency_histogram') and 'latency' in name:
            self.latency_histogram.record(value, labels)
            
        # Add to buffer
        metric_point = MetricPoint(
            timestamp=datetime.now(),
            name=name,
            value=value,
            unit="seconds" if "latency" in name or "duration" in name else "value",
            labels=labels,
            description=f"Histogram metric: {name}"
        )
        self.metrics_buffer.append(metric_point)
        
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record gauge metric"""
        labels = labels or {}
        
        # Record in internal storage
        self.gauges[name] = value
        
        # Record in OpenTelemetry
        if hasattr(self, 'model_accuracy') and 'accuracy' in name:
            self.model_accuracy.add(value, labels)
            
        # Add to buffer
        metric_point = MetricPoint(
            timestamp=datetime.now(),
            name=name,
            value=value,
            unit="percent" if "percent" in name else "bytes" if "bytes" in name else "value",
            labels=labels,
            description=f"Gauge metric: {name}"
        )
        self.metrics_buffer.append(metric_point)
        
    def get_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for the specified time window"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        # Filter metrics by time window
        recent_metrics = [m for m in self.metrics_buffer if m.timestamp > cutoff_time]
        
        # Group by metric name
        grouped_metrics = defaultdict(list)
        for metric in recent_metrics:
            grouped_metrics[metric.name].append(metric.value)
            
        # Calculate summary statistics
        summary = {}
        for name, values in grouped_metrics.items():
            if values:
                summary[name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'latest': values[-1]
                }
                
                # Add percentiles for histograms
                if len(values) > 1:
                    sorted_values = sorted(values)
                    summary[name].update({
                        'p50': sorted_values[len(sorted_values) // 2],
                        'p95': sorted_values[int(len(sorted_values) * 0.95)],
                        'p99': sorted_values[int(len(sorted_values) * 0.99)]
                    })
                    
        return summary

class DistributedTracer:
    """Distributed tracing with OpenTelemetry"""
    
    def __init__(self, service_name: str, config: Dict[str, Any] = None):
        self.service_name = service_name
        self.config = config or {}
        
        # Setup OpenTelemetry tracing
        self._setup_otel_tracing()
        
        # Span storage
        self.spans = {}
        self.completed_spans = deque(maxlen=1000)
        
    def _setup_otel_tracing(self):
        """Setup OpenTelemetry tracing"""
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0",
            "deployment.environment": os.getenv('ENVIRONMENT', 'development')
        })
        
        trace.set_tracer_provider(TracerProvider(resource=resource))
        
        # Setup Jaeger exporter if configured
        if self.config.get('jaeger', {}).get('enabled', False):
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config['jaeger'].get('host', 'localhost'),
                agent_port=self.config['jaeger'].get('port', 6831),
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
        self.tracer = trace.get_tracer(__name__)
        
    def start_span(self, operation_name: str, parent_span_id: str = None, **tags) -> str:
        """Start a new tracing span"""
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add tags
            for key, value in tags.items():
                span.set_attribute(key, str(value))
                
            span_context = span.get_span_context()
            span_id = format(span_context.span_id, '016x')
            trace_id = format(span_context.trace_id, '032x')
            
            # Store span information
            trace_span = TraceSpan(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                operation_name=operation_name,
                start_time=datetime.now(),
                end_time=None,
                duration_ms=None,
                status="active",
                tags=tags
            )
            
            self.spans[span_id] = trace_span
            return span_id
            
    def finish_span(self, span_id: str, status: str = "success", **tags):
        """Finish a tracing span"""
        if span_id in self.spans:
            span = self.spans[span_id]
            span.end_time = datetime.now()
            span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
            span.status = status
            span.tags.update(tags)
            
            # Move to completed spans
            self.completed_spans.append(span)
            del self.spans[span_id]
            
    def add_span_log(self, span_id: str, message: str, **fields):
        """Add log entry to span"""
        if span_id in self.spans:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'message': message,
                **fields
            }
            self.spans[span_id].logs.append(log_entry)
            
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary for a complete trace"""
        trace_spans = [s for s in self.completed_spans if s.trace_id == trace_id]
        
        if not trace_spans:
            return {}
            
        # Calculate trace statistics
        total_duration = max(s.duration_ms for s in trace_spans if s.duration_ms)
        span_count = len(trace_spans)
        error_count = len([s for s in trace_spans if s.status == "error"])
        
        return {
            'trace_id': trace_id,
            'total_duration_ms': total_duration,
            'span_count': span_count,
            'error_count': error_count,
            'success_rate': (span_count - error_count) / span_count if span_count > 0 else 0,
            'spans': [asdict(s) for s in trace_spans]
        }

class MLOpsObservability:
    """Comprehensive observability system for MLOps"""
    
    def __init__(self, service_name: str, config_path: str = "observability_config.json"):
        self.service_name = service_name
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.logger = StructuredLogger(service_name, self.config.get('logging', {}))
        self.metrics = MetricsCollector(self.config.get('metrics', {}))
        self.tracer = DistributedTracer(service_name, self.config.get('tracing', {}))
        
        # Initialize OpenTelemetry instrumentation
        LoggingInstrumentor().instrument(set_logging_format=True)
        
        self.logger.info("MLOps Observability system initialized", 
                        service=service_name, config=self.config)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load observability configuration"""
        default_config = {
            "logging": {
                "level": "INFO",
                "file_logging": {
                    "enabled": True,
                    "path": "logs/mlops.log",
                    "max_size_mb": 100,
                    "backup_count": 5
                }
            },
            "metrics": {
                "system_metrics": True,
                "prometheus": {
                    "enabled": True,
                    "port": 8000
                }
            },
            "tracing": {
                "jaeger": {
                    "enabled": False,
                    "host": "localhost",
                    "port": 6831
                }
            },
            "alerting": {
                "enabled": True,
                "thresholds": {
                    "error_rate": 0.05,
                    "latency_p95": 1000,
                    "cpu_usage": 80,
                    "memory_usage": 85
                }
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge configurations
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                        
        return default_config
        
    def observe_function(self, operation_name: str = None, 
                        log_args: bool = False, 
                        log_result: bool = False):
        """Decorator to add observability to functions"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                # Start span
                span_id = self.tracer.start_span(
                    op_name,
                    function=func.__name__,
                    module=func.__module__
                )
                
                # Set logging context
                self.logger.set_context(
                    operation=op_name,
                    span_id=span_id
                )
                
                start_time = time.time()
                
                try:
                    # Log function entry
                    log_data = {'operation': op_name}
                    if log_args:
                        log_data['args'] = str(args)
                        log_data['kwargs'] = str(kwargs)
                        
                    self.logger.info(f"Starting {op_name}", **log_data)
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Calculate duration
                    duration = time.time() - start_time
                    
                    # Record metrics
                    self.metrics.record_counter(f"{op_name}.calls")
                    self.metrics.record_histogram(f"{op_name}.duration", duration)
                    
                    # Log success
                    log_data = {'operation': op_name, 'duration_seconds': duration}
                    if log_result:
                        log_data['result'] = str(result)
                        
                    self.logger.info(f"Completed {op_name}", **log_data)
                    
                    # Finish span
                    self.tracer.finish_span(span_id, "success", duration_ms=duration*1000)
                    
                    return result
                    
                except Exception as e:
                    # Calculate duration
                    duration = time.time() - start_time
                    
                    # Record error metrics
                    self.metrics.record_counter(f"{op_name}.errors")
                    self.metrics.record_histogram(f"{op_name}.duration", duration)
                    
                    # Log error
                    self.logger.error(f"Error in {op_name}: {str(e)}", 
                                    operation=op_name, 
                                    duration_seconds=duration,
                                    error_type=type(e).__name__)
                    
                    # Finish span with error
                    self.tracer.finish_span(span_id, "error", 
                                          duration_ms=duration*1000,
                                          error_message=str(e))
                    
                    raise
                    
                finally:
                    # Clear logging context
                    self.logger.clear_context()
                    
            return wrapper
        return decorator
        
    def observe_model_prediction(self, model_name: str, version: str):
        """Decorator specifically for model prediction functions"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                span_id = self.tracer.start_span(
                    "model.prediction",
                    model_name=model_name,
                    model_version=version
                )
                
                self.logger.set_context(
                    model_name=model_name,
                    model_version=version,
                    span_id=span_id
                )
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Record prediction metrics
                    self.metrics.record_counter("model.predictions.total", 
                                              labels={"model": model_name, "version": version})
                    self.metrics.record_histogram("model.prediction.latency", duration,
                                                labels={"model": model_name, "version": version})
                    
                    self.logger.info("Model prediction completed", 
                                   prediction_latency=duration,
                                   model_name=model_name,
                                   model_version=version)
                    
                    self.tracer.finish_span(span_id, "success", 
                                          prediction_latency=duration)
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    self.metrics.record_counter("model.predictions.errors", 
                                              labels={"model": model_name, "version": version})
                    
                    self.logger.error("Model prediction failed", 
                                    error=str(e),
                                    model_name=model_name,
                                    model_version=version)
                    
                    self.tracer.finish_span(span_id, "error", error_message=str(e))
                    
                    raise
                    
                finally:
                    self.logger.clear_context()
                    
            return wrapper
        return decorator
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            "service": self.service_name,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self._start_time if hasattr(self, '_start_time') else 0,
            "metrics_summary": self.metrics.get_metrics_summary(),
            "active_spans": len(self.tracer.spans),
            "total_spans": len(self.tracer.completed_spans),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "disk_total": psutil.disk_usage('/').total,
                "hostname": socket.gethostname(),
                "platform": sys.platform
            }
        }
        
    def export_traces(self, trace_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Export trace data"""
        if trace_ids:
            return [self.tracer.get_trace_summary(trace_id) for trace_id in trace_ids]
        else:
            # Export recent traces
            unique_trace_ids = set(s.trace_id for s in self.tracer.completed_spans)
            return [self.tracer.get_trace_summary(trace_id) for trace_id in unique_trace_ids]

def create_observability_config():
    """Create sample observability configuration"""
    config = {
        "logging": {
            "level": "INFO",
            "file_logging": {
                "enabled": True,
                "path": "logs/mlops.log",
                "max_size_mb": 100,
                "backup_count": 5
            }
        },
        "metrics": {
            "system_metrics": True,
            "prometheus": {
                "enabled": True,
                "port": 8000
            }
        },
        "tracing": {
            "jaeger": {
                "enabled": True,
                "host": "localhost",
                "port": 6831
            }
        },
        "alerting": {
            "enabled": True,
            "thresholds": {
                "error_rate": 0.05,
                "latency_p95": 1000,
                "cpu_usage": 80,
                "memory_usage": 85
            }
        }
    }
    
    with open("observability_config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    print("Created observability_config.json")

# Example usage decorators
def with_observability(observability: MLOpsObservability):
    """Context manager for observability"""
    class ObservabilityContext:
        def __init__(self, obs):
            self.obs = obs
            
        def __enter__(self):
            self.obs._start_time = time.time()
            return self.obs
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                self.obs.logger.error("Unhandled exception", 
                                    exception_type=exc_type.__name__,
                                    exception_message=str(exc_val))
                                    
    return ObservabilityContext(observability)

if __name__ == "__main__":
    # Create sample configuration
    create_observability_config()
    
    # Example usage
    obs = MLOpsObservability("mlops-pipeline")
    
    @obs.observe_function("example.calculation", log_args=True, log_result=True)
    def example_calculation(x: int, y: int) -> int:
        time.sleep(0.1)  # Simulate work
        return x + y
        
    @obs.observe_model_prediction("iris_classifier", "v1.0")
    def predict_iris(features):
        time.sleep(0.05)  # Simulate prediction
        return "setosa"
        
    # Test the functions
    result = example_calculation(5, 3)
    prediction = predict_iris([5.1, 3.5, 1.4, 0.2])
    
    # Get health status
    health = obs.get_health_status()
    print(f"Health status: {json.dumps(health, indent=2)}")
    
    # Get metrics summary
    metrics = obs.metrics.get_metrics_summary()
    print(f"Metrics summary: {json.dumps(metrics, indent=2)}")