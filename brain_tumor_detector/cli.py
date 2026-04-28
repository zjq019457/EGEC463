import click
from pathlib import Path
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .utils import download_dataset


@click.command()
@click.option('--config', default='config/default.yaml', help='Path to config file.')
@click.option('--skip-eval', is_flag=True, default=False, help='Skip evaluation after training.')
@click.option('--eval-output', default='runs/evaluate', help='Directory to save evaluation figures.')
def main(config, skip_eval, eval_output):
    """Main training pipeline."""
    download_dataset(config)
    trainer = ModelTrainer(config)
    trainer.train()

    if skip_eval:
        click.echo('Skipping evaluation as requested.')
        return

    best_model_path = Path(trainer.config['model']['save_dir']) / 'weights' / 'best.pt'
    if not best_model_path.exists():
        raise FileNotFoundError(f'Best model not found at {best_model_path}')

    evaluator = ModelEvaluator(str(best_model_path))
    validation_dir = Path(trainer.config['data']['validation_images_dir'])
    label_dir = Path(trainer.config['data']['validation_labels_dir'])

    validation_pairs = []
    for image_file in sorted(validation_dir.glob('*')):
        if image_file.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue
        label_file = label_dir / f'{image_file.stem}.txt'
        if label_file.exists():
            validation_pairs.append((str(image_file), str(label_file)))

    if not validation_pairs:
        raise FileNotFoundError('No validation images found for evaluation.')

    evaluator.evaluate(validation_pairs, output_dir=eval_output)
    click.echo(f'Evaluation images saved to: {eval_output}')
