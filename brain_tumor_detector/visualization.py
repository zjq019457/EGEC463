import matplotlib.pyplot as plt
from pathlib import Path

def plot_comparison(original, prediction, title, save_path: Path | str | None = None):
    """Plot original image vs model prediction and optionally save to disk."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(prediction)
    ax2.set_title('Model Prediction')
    ax2.axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
