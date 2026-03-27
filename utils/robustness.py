# GenAI usage statement: Claude (Anthropic) was used in an assistive role to help
# structure and refine parts of this robustness evaluation module. All implementation
# details, experimental design, and analysis choices are the author's own.
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from PIL import Image
import json
import math
import time
from utils.common import evaluate_model


class NoisyDataset(Dataset):
    """
    Dataset wrapper that injects Gaussian noise into input images.

    Used to evaluate model robustness to input perturbations by applying
    additive noise at inference time.

    Args:
        dataset (Dataset): Base dataset (e.g. CIFAR-10).
        noise_std (float): Standard deviation of Gaussian noise.

    Notes:
        - Noise is applied per sample on-the-fly.
        - Output is clamped to the normalised input range [-1, 1].
    """
    def __init__(self, dataset, noise_std=0.1):
        self.dataset = dataset
        self.noise_std = noise_std
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        noise = torch.randn_like(x) * self.noise_std
        x_noisy = x + noise
        # keep values in normalised range
        x_noisy = torch.clamp(x_noisy, -1.0, 1.0)
        return x_noisy, y


def build_noisy_test_loader(test_dataset, batch_size, noise_std=0.1):
    """
        Construct a DataLoader with Gaussian noise applied to inputs.

        Args:
            test_dataset (Dataset): Clean test dataset.
            batch_size (int): Batch size for evaluation.
            noise_std (float): Standard deviation of Gaussian noise.

        Returns:
            DataLoader: Noisy test loader.
        """
    noisy_dataset = NoisyDataset(test_dataset, noise_std)
    return DataLoader(noisy_dataset, batch_size=batch_size, shuffle=False)


def save_mixup_demo(mixup_fn, dataset, save_path, alpha=0.4, device="cpu"):
    """
    Generate and save a visual demonstration of MixUp augmentation.

    Creates a 4x4 grid of mixed images by applying MixUp to sampled inputs.
    Useful for qualitative inspection of interpolation behaviour.

    Args:
        mixup_fn (callable): MixUp function returning mixed inputs and labels.
        dataset (Dataset): Source dataset (e.g. CIFAR-10).
        save_path (Path | str): File path to save the image.
        alpha (float): MixUp Beta distribution parameter.
        device (str | torch.device): Device for computation.

    Output:
        Saves an image file showing mixed samples.
    """
    import numpy as np
    samples = torch.stack([dataset[i][0] for i in range(16)]).to(device)
    labels = torch.tensor([dataset[i][1] for i in range(16)]).to(device)
    mixed, _, _, _ = mixup_fn(samples, labels, alpha)
    # denormalise CIFAR10 for visualisation
    mixed = (mixed * 0.5) + 0.5
    mixed = mixed.clamp(0, 1)
    grid = torch.zeros(3, 32 * 4, 32 * 4)
    idx = 0
    for r in range(4):
        for c in range(4):
            grid[:, r*32:(r+1)*32, c*32:(c+1)*32] = mixed[idx]
            idx += 1
    img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)
    print(f"MixUp demo saved to: {save_path}")

def evaluate_noise_robustness(model, test_dataset, batch_size, save_path, noise_levels=None):
    """
    Evaluate model performance across multiple levels of Gaussian noise.

    This function measures how accuracy degrades as input noise increases,
    providing a robustness profile rather than a single-point estimate.

    Args:
        model (torch.nn.Module): Trained model (evaluation mode assumed).
        test_dataset (Dataset): Clean test dataset.
        batch_size (int): Batch size for evaluation.
        save_path (Path | str): File path to save results (JSON).
        noise_levels (list[float] | None): Noise standard deviations to evaluate.

    Returns:
        dict: Mapping of noise_std (float) to accuracy (float).

    Side effects:
        - Prints accuracy for each noise level
        - Saves results to JSON file
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    criterion = nn.CrossEntropyLoss()
    results = {}
    print("\n--- Noise Robustness ---")
    for std in noise_levels:
        loader = build_noisy_test_loader(test_dataset, batch_size, noise_std=std)
        _, acc = evaluate_model(loader, model, criterion)
        print(f"  noise_std={std:.2f}  accuracy={acc:.4f}")
        results[std] = acc
    # save noise results to JSON
    payload = {
        "stage": "noise_test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "noise_levels": noise_levels,
            "accuracy_by_noise": {str(k): v for k, v in results.items()}
        }
    }
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"Noise robustness results saved to: {save_path}")
    return results