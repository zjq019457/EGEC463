import cv2
from pathlib import Path
from .models import load_model
from .visualization import plot_comparison


class ModelEvaluator:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def evaluate(self, validation_pairs, output_dir: str = 'runs/evaluate'):
        """Evaluate model on validation image pairs and save comparison figures."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for img_path, _ in validation_pairs:
            img_path = Path(img_path)
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.model(str(img_path))
            prediction = results[0].plot()
            prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)

            plot_comparison(
                image,
                prediction,
                title=str(img_path.name),
                save_path=output_path / f"{img_path.stem}_comparison.png",
            )
