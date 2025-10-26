#!/usr/bin/env python3
"""
Global delay tracker for AI API requests.

This module provides a global mechanism to track and enforce delays between
AI API requests across all AI service instances to avoid connection issues.
"""

import time
import threading
import os
import logging

class GlobalDelayTracker:
    """
    Global delay tracker to enforce delays between AI API requests.
    
    This class uses a singleton pattern to ensure all AI service instances
    share the same delay tracking mechanism.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GlobalDelayTracker, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the global delay tracker."""
        if self._initialized:
            return
            
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_request_time = 0
        self.request_lock = threading.Lock()
        
        # Note: Delay configuration is read dynamically from environment
        # to ensure it picks up changes from .env file loading

        # Initialize with current environment value
        self._update_delay_from_env()

        if self.request_delay > 0:
            print(f"🕐 Global AI request delay initialized: {self.request_delay}s")
            self.logger.info(f"Global AI request delay initialized: {self.request_delay}s")
        
        self._initialized = True

    def _update_delay_from_env(self):
        """Update delay setting from environment variable."""
        self.request_delay = float(os.getenv('AI_REQUEST_DELAY', '0'))

    def apply_delay(self, service_name: str = "AI"):
        """
        Apply delay before making an AI API request.

        Args:
            service_name: Name of the AI service making the request (for logging)
        """
        # Refresh delay setting from environment each time
        self._update_delay_from_env()

        if self.request_delay <= 0:
            return
        
        with self.request_lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            
            if time_since_last_request < self.request_delay:
                delay_needed = self.request_delay - time_since_last_request
                print(f"🕐 [{service_name}] Applying {delay_needed:.1f}s delay before next AI request...")
                self.logger.info(f"[{service_name}] Applying {delay_needed:.1f}s delay before next AI request...")
                time.sleep(delay_needed)
            
            self.last_request_time = time.time()
    
    def get_delay_setting(self) -> float:
        """Get the current delay setting."""
        return self.request_delay
    
    def set_delay(self, delay_seconds: float):
        """
        Set a new delay value.
        
        Args:
            delay_seconds: New delay in seconds
        """
        with self.request_lock:
            self.request_delay = delay_seconds
            self.logger.info(f"Global AI request delay updated: {self.request_delay}s")


# Global instance
global_delay_tracker = GlobalDelayTracker()
