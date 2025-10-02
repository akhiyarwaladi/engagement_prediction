"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path: Project root directory
    """
    # Assuming this file is in src/utils/, go up 2 levels
    return Path(__file__).parent.parent.parent


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config.yaml

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if config_path is None:
        config_path = get_project_root() / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_data_path(filename: str, data_type: str = 'raw') -> Path:
    """Get full path for data file.

    Args:
        filename: Name of the data file
        data_type: Type of data ('raw', 'processed', 'features', 'academic')

    Returns:
        Full path to the data file
    """
    root = get_project_root()

    if data_type == 'raw':
        return root / filename
    else:
        return root / 'data' / data_type / filename


def get_model_path(model_name: str) -> Path:
    """Get full path for model file.

    Args:
        model_name: Name of the model file

    Returns:
        Full path to the model file
    """
    return get_project_root() / 'models' / model_name
