# This file is part of airgapped-offline-rag.
#
# Airgapped Offline RAG is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Airgapped Offline RAG is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Airgapped Offline RAG. If not, see <https://www.gnu.org/licenses/>.
#
# Copyright (C) 2024 Vincent Koc (https://github.com/vincentkoc)

import os
import time
import logging
import functools
from typing import Any, Dict, Optional, Callable
import streamlit as st

try:
    from opik import Opik, track
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    track = lambda **kwargs: lambda func: func  # No-op decorator

logger = logging.getLogger(__name__)

class TelemetryManager:
    """Manages telemetry and observability for the RAG application"""
    
    def __init__(self):
        self.enabled = self._should_enable_telemetry()
        self.client = None
        self._initialize_opik()
    
    def _should_enable_telemetry(self) -> bool:
        """Check if telemetry should be enabled based on environment"""
        if not OPIK_AVAILABLE:
            logger.warning("Opik not available. Telemetry disabled.")
            return False
            
        enable_telemetry = os.getenv('ENABLE_TELEMETRY', 'True').lower() == 'true'
        has_api_key = bool(os.getenv('OPIK_API_KEY'))
        
        if enable_telemetry and not has_api_key:
            logger.warning("ENABLE_TELEMETRY=True but OPIK_API_KEY not found. Telemetry disabled.")
            return False
            
        return enable_telemetry
    
    def _initialize_opik(self):
        """Initialize Opik client if telemetry is enabled"""
        if not self.enabled:
            return
            
        try:
            project_name = os.getenv('OPIK_PROJECT_NAME', 'airgapped-offline-rag')
            self.client = Opik(project_name=project_name)
            logger.info(f"Opik telemetry initialized for project: {project_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Opik: {e}")
            self.enabled = False
    
    def track_model_inference(self, model_name: str, input_tokens: int = 0, output_tokens: int = 0):
        """Decorator to track model inference performance"""
        def decorator(func: Callable) -> Callable:
            if not self.enabled:
                return func
                
            @functools.wraps(func)
            @track(
                name=f"model_inference_{model_name}",
                tags=["inference", model_name],
                metadata={"model_name": model_name}
            )
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Log success metrics
                    inference_time = time.time() - start_time
                    self._log_inference_metrics(
                        model_name=model_name,
                        inference_time=inference_time,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        success=True
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log error metrics
                    inference_time = time.time() - start_time
                    self._log_inference_metrics(
                        model_name=model_name,
                        inference_time=inference_time,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        success=False,
                        error=str(e)
                    )
                    raise
                    
            return wrapper
        return decorator
    
    def track_rag_operation(self, operation_type: str):
        """Decorator to track RAG pipeline operations"""
        def decorator(func: Callable) -> Callable:
            if not self.enabled:
                return func
                
            @functools.wraps(func)
            @track(
                name=f"rag_{operation_type}",
                tags=["rag", operation_type],
                metadata={"operation_type": operation_type}
            )
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Log success metrics
                    operation_time = time.time() - start_time
                    self._log_rag_metrics(
                        operation_type=operation_type,
                        operation_time=operation_time,
                        success=True,
                        result_info=self._extract_result_info(result)
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log error metrics
                    operation_time = time.time() - start_time
                    self._log_rag_metrics(
                        operation_type=operation_type,
                        operation_time=operation_time,
                        success=False,
                        error=str(e)
                    )
                    raise
                    
            return wrapper
        return decorator
    
    def track_document_processing(self, doc_format: str, doc_size: int = 0):
        """Decorator to track document processing performance"""
        def decorator(func: Callable) -> Callable:
            if not self.enabled:
                return func
                
            @functools.wraps(func)
            @track(
                name=f"document_processing_{doc_format}",
                tags=["document_processing", doc_format],
                metadata={"doc_format": doc_format, "doc_size": doc_size}
            )
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Log success metrics
                    processing_time = time.time() - start_time
                    self._log_document_metrics(
                        doc_format=doc_format,
                        doc_size=doc_size,
                        processing_time=processing_time,
                        success=True,
                        chunks_created=len(result) if isinstance(result, list) else 1
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log error metrics
                    processing_time = time.time() - start_time
                    self._log_document_metrics(
                        doc_format=doc_format,
                        doc_size=doc_size,
                        processing_time=processing_time,
                        success=False,
                        error=str(e)
                    )
                    raise
                    
            return wrapper
        return decorator
    
    def _log_inference_metrics(self, model_name: str, inference_time: float, 
                              input_tokens: int, output_tokens: int, success: bool, 
                              error: Optional[str] = None):
        """Log model inference metrics"""
        metrics = {
            "model_name": model_name,
            "inference_time_seconds": inference_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "tokens_per_second": (input_tokens + output_tokens) / inference_time if inference_time > 0 else 0,
            "success": success
        }
        
        if error:
            metrics["error"] = error
            
        logger.info(f"Model inference metrics: {metrics}")
    
    def _log_rag_metrics(self, operation_type: str, operation_time: float, 
                        success: bool, result_info: Optional[Dict] = None, 
                        error: Optional[str] = None):
        """Log RAG operation metrics"""
        metrics = {
            "operation_type": operation_type,
            "operation_time_seconds": operation_time,
            "success": success
        }
        
        if result_info:
            metrics.update(result_info)
            
        if error:
            metrics["error"] = error
            
        logger.info(f"RAG operation metrics: {metrics}")
    
    def _log_document_metrics(self, doc_format: str, doc_size: int, 
                             processing_time: float, success: bool, 
                             chunks_created: int = 0, error: Optional[str] = None):
        """Log document processing metrics"""
        metrics = {
            "doc_format": doc_format,
            "doc_size_bytes": doc_size,
            "processing_time_seconds": processing_time,
            "processing_speed_bytes_per_second": doc_size / processing_time if processing_time > 0 else 0,
            "chunks_created": chunks_created,
            "success": success
        }
        
        if error:
            metrics["error"] = error
            
        logger.info(f"Document processing metrics: {metrics}")
    
    def _extract_result_info(self, result: Any) -> Dict[str, Any]:
        """Extract useful information from operation results"""
        info = {}
        
        if isinstance(result, str):
            info["result_length"] = len(result)
        elif isinstance(result, list):
            info["result_count"] = len(result)
            if result and isinstance(result[0], str):
                info["avg_result_length"] = sum(len(r) for r in result) / len(result)
        elif isinstance(result, dict):
            info["result_keys"] = list(result.keys())
            
        return info
    
    def log_user_interaction(self, interaction_type: str, metadata: Optional[Dict] = None):
        """Log user interactions for analytics"""
        if not self.enabled:
            return
            
        try:
            interaction_data = {
                "interaction_type": interaction_type,
                "timestamp": time.time(),
                "session_id": self._get_session_id()
            }
            
            if metadata:
                interaction_data.update(metadata)
                
            logger.info(f"User interaction: {interaction_data}")
            
        except Exception as e:
            logger.error(f"Failed to log user interaction: {e}")
    
    def _get_session_id(self) -> str:
        """Get or create a session ID for this Streamlit session"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(int(time.time() * 1000))
        return st.session_state.session_id
    
    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """Log errors for monitoring"""
        if not self.enabled:
            return
            
        try:
            error_data = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp": time.time(),
                "session_id": self._get_session_id()
            }
            
            if context:
                error_data.update(context)
                
            logger.error(f"Application error: {error_data}")
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")

# Global telemetry manager instance
telemetry = TelemetryManager()

# Convenience decorators
track_inference = telemetry.track_model_inference
track_rag = telemetry.track_rag_operation
track_document_processing = telemetry.track_document_processing