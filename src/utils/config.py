"""
Configuration Loader for Vital Signs Monitoring System
Loads settings from config.yaml and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Camera configuration settings"""
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    type: str = "mysql"
    host: str = "localhost"
    port: int = 3306
    name: str = "vitals_monitoring"
    user: str = "root"
    password: str = ""


@dataclass
class ApiConfig:
    """FastAPI configuration settings"""
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True


class Config:
    """
    Central configuration manager.
    Loads from config.yaml with environment variable overrides.
    
    Usage:
        config = Config()
        camera_id = config.get("camera.device_id")
        # Or use typed configs
        camera = config.camera
    """
    
    _instance: Optional['Config'] = None
    _config: dict = {}
    
    def __new__(cls) -> 'Config':
        """Singleton pattern - only one config instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        # Find project root (where config.yaml lives)
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        config_path = project_root / "config.yaml"
        
        if not config_path.exists():
            print(f"Warning: config.yaml not found at {config_path}")
            self._config = {}
            return
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f) or {}
        
        # Override with environment variables
        self._apply_env_overrides()
    
    def _apply_env_overrides(self) -> None:
        """Override config values with environment variables"""
        # Database password from env
        if os.getenv("DB_PASSWORD"):
            self._set_nested("database.password", os.getenv("DB_PASSWORD"))
        
        # Supabase credentials from env
        if os.getenv("SUPABASE_URL"):
            self._set_nested("database.supabase_url", os.getenv("SUPABASE_URL"))
        if os.getenv("SUPABASE_KEY"):
            self._set_nested("database.supabase_key", os.getenv("SUPABASE_KEY"))
    
    def _set_nested(self, key: str, value: Any) -> None:
        """Set a nested configuration value using dot notation"""
        keys = key.split(".")
        d = self._config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Dot-separated key path (e.g., "camera.width")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    @property
    def camera(self) -> CameraConfig:
        """Get typed camera configuration"""
        cam = self.get("camera", {})
        return CameraConfig(
            device_id=cam.get("device_id", 0),
            width=cam.get("width", 1280),
            height=cam.get("height", 720),
            fps=cam.get("fps", 30)
        )
    
    @property
    def database(self) -> DatabaseConfig:
        """Get typed database configuration"""
        db = self.get("database", {})
        return DatabaseConfig(
            type=db.get("type", "mysql"),
            host=db.get("host", "localhost"),
            port=db.get("port", 3306),
            name=db.get("name", "vitals_monitoring"),
            user=db.get("user", "root"),
            password=db.get("password", "")
        )
    
    @property
    def api(self) -> ApiConfig:
        """Get typed API configuration"""
        api = self.get("api", {})
        return ApiConfig(
            host=api.get("host", "127.0.0.1"),
            port=api.get("port", 8000),
            debug=api.get("debug", True)
        )
    
    @property
    def project_root(self) -> Path:
        """Get the project root directory"""
        return Path(__file__).resolve().parent.parent.parent
    
    def reload(self) -> None:
        """Reload configuration from file"""
        self._load_config()


# Convenience function for quick access
def get_config() -> Config:
    """Get the singleton Config instance"""
    return Config()


if __name__ == "__main__":
    # Test the configuration loader
    config = Config()
    print(f"Project: {config.get('project.name')}")
    print(f"Camera: {config.camera}")
    print(f"Database: {config.database}")
    print(f"API: {config.api}")
