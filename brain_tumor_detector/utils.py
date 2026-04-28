import urllib.request
from pathlib import Path
import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def download_dataset(config_path: str = 'config/default.yaml') -> Path:
    """Download the brain tumor dataset YAML file if needed."""
    config = load_config(config_path)
    data_config = config.get('data', {})
    url = data_config.get('yaml_url')
    local_path = Path(data_config.get('local_path', 'brain-tumor.yaml'))

    if local_path.exists():
        return local_path

    if not url:
        raise ValueError('Missing data.yaml_url in config file.')

    local_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, local_path)

    if not local_path.exists():
        raise RuntimeError(f'Dataset YAML download failed: {local_path}')

    return local_path
