# GenAI usage statement: Claude (Anthropic) was used in an assistive role to help
# structure and debug this file. All deep learning logic, analysis, and design
# decisions are the author's own.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from utils.common import *
from utils.robustness import build_noisy_test_loader, save_mixup_demo, evaluate_noise_robustness



TASK_DIR = Path(__file__).resolve().parent
MODEL_DIR = TASK_DIR / "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_analysis(history, noisy_test_metrics, noise_results):
    """
    Print quantitative summary statistics to support the written technical analysis
    for Task 2.

    Args:
        history (dict): Training history loaded from JSON.
        noisy_test_metrics (dict): Metrics from evaluation on the noisy test set.
        noise_results (dict): Accuracy results across multiple noise levels.
    """
    metrics = history["metrics"]["epoch_metrics"]
    train_acc = [m.get("train_accuracy") for m in metrics]
    val_acc = [m["validation_accuracy"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    val_loss = [m["validation_loss"] for m in metrics]

    best_val_acc = max(val_acc) if val_acc else None
    best_val_epoch = val_acc.index(best_val_acc) + 1 if val_acc else None
    final_train_acc = train_acc[-1] if train_acc else None
    final_val_acc = val_acc[-1] if val_acc else None

    print("=" * 60)
    print("TASK 2 — QUANTITATIVE ANALYSIS SUMMARY")
    print("=" * 60)

    print("\n--- Training Summary ---")
    if final_train_acc is not None:
        print(f"  Final train accuracy      : {final_train_acc:.4f}")
    if final_val_acc is not None:
        print(f"  Final validation accuracy : {final_val_acc:.4f}")
    if best_val_acc is not None:
        print(f"  Peak validation accuracy  : {best_val_acc:.4f} (epoch {best_val_epoch})")

    print("\n--- Noisy Test Performance ---")
    print(f"  Test Loss     : {noisy_test_metrics['test_loss']:.4f}")
    print(f"  Test Accuracy : {noisy_test_metrics['test_accuracy']:.4f}")

    print("\n--- Noise Robustness Curve ---")
    for std, acc in noise_results.items():
        print(f"  noise_std={float(std):.2f}  accuracy={acc:.4f}")

    if noise_results:
        noise_keys = sorted(float(k) for k in noise_results.keys())
        first_key = str(noise_keys[0])
        last_key = str(noise_keys[-1])

        # handle cases like "0" vs "0.0"
        if first_key not in noise_results:
            first_key = min(noise_results.keys(), key=lambda x: float(x))
        if last_key not in noise_results:
            last_key = max(noise_results.keys(), key=lambda x: float(x))

        start_acc = noise_results[first_key]
        end_acc = noise_results[last_key]

        print("\n--- Robustness Degradation ---")
        print(f"  Accuracy at lowest noise  : {start_acc:.4f}")
        print(f"  Accuracy at highest noise : {end_acc:.4f}")
        print(f"  Total drop                : {start_acc - end_acc:.4f}")

    print("\n--- Technical Analysis ---")
    print("""
[REPLACE THIS BLOCK WITH YOUR TASK 2 ANALYSIS COVERING:]
  1. Why MixUp reduces memorisation and encourages smoother decision boundaries
  2. Why label smoothing reduces overconfidence and overshooting
  3. How early stopping prevented further validation degradation
  4. What the noisy test and robustness curve show about generalisation
  5. Whether robustness degrades gradually or sharply as noise increases
    """)
    print("=" * 60)

def evaluate_noisy_test(model, test_dataset, batch_size, name, config):
    """
    Evaluate trained model on noisy test data.
    """
    test_loader = build_noisy_test_loader(test_dataset, batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(test_loader, model, criterion)
    print("\nNoisy test performance")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    test_metrics = {"test_loss": test_loss,"test_accuracy": test_acc}
    history_path = save_history(test_metrics,name,"noisy_test",model,MODEL_DIR,config=config)
    return test_metrics, history_path



def main():
    # Load training history
    history = load_history(MODEL_DIR / "baseline_mixup_smooth_train_history.json")
    config = history["config"]
    batch_size = config["batch_size"]
    # Load trained model
    model = load_model(dropout_prob=config.get("dropout_prob", 0.0), weights_path=MODEL_DIR / "baseline_mixup_smooth_model.pt")
    print("Model loaded:", type(model).__name__)
    # Load CIFAR10
    _, test_dataset = download_data()
    # Evaluate on noisy test set
    noisy_test_metrics, _ = evaluate_noisy_test(model,test_dataset,batch_size,"baseline_mixup_smooth",config)
    # noise robustness curve
    noise_results = evaluate_noise_robustness(model, test_dataset, batch_size, TASK_DIR / "noise_robustness.json")
    # Generate MixUp demo figure
    save_mixup_demo(mixup_data,test_dataset,TASK_DIR / "robustness_demo.png",alpha=config.get("mixup_alpha", 0.4),
                    device=device)
    # Print analysis summary
    print_analysis(history, noisy_test_metrics, noise_results)
    summary = {
        "config": config,
        "noisy_test_metrics": noisy_test_metrics,
        "noise_robustness_curve": noise_results
    }
    save_json(summary, TASK_DIR / "task2_summary.json")


if __name__ == "__main__":
    main()