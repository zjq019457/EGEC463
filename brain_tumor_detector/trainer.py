from pathlib import Path
import torch
from .models import load_model
from .utils import load_config


class ModelTrainer:
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = load_config(config_path)
        self.model = load_model(self.config['model']['size'])

    def train(self):
        """Train the model using configuration settings."""
        data_config = self.config['data']
        device = self.config['training'].get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        training_args = {
            'data': str(Path(data_config['local_path']).resolve()),
            'epochs': self.config['training']['epochs'],
            'batch': self.config['training']['batch_size'],
            'imgsz': self.config['training']['image_size'],
            'save_dir': self.config['model']['save_dir'],
            'device': device,
            **self.config.get('augmentation', {}),
        }

        return self.model.train(**training_args)
