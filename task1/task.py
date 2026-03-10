# GenAI usage statement: Claude (Anthropic) was used in an assistive role to help
# structure and debug this file. All deep learning logic, analysis, and design
# decisions are the author's own.

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from utils.plotting import generate_gap_plot, generate_gap_per_epoch_plot
from utils.common import *


TASK_DIR   = Path(__file__).resolve().parent
MODEL_DIR  = TASK_DIR / "models"
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_analysis(b_epochs, b_train_acc, b_val_acc, r_epochs, r_train_acc, r_val_acc, base_conf,
                   reg_conf, base_test_metrics, reg_test_metrics):
    """
    Print quantitative summary statistics to support the written technical analysis.

    Args:
        b_epochs    (list[int]):   Baseline epoch numbers.
        b_train_acc (list[float]): Baseline training accuracy per epoch.
        b_val_acc   (list[float]): Baseline validation accuracy per epoch.
        r_epochs    (list[int]):   Regularised epoch numbers.
        r_train_acc (list[float]): Regularised training accuracy per epoch.
        r_val_acc   (list[float]): Regularised validation accuracy per epoch.
    """
    b_gap_final = b_train_acc[-1] - b_val_acc[-1]
    r_gap_final = r_train_acc[-1] - r_val_acc[-1]

    b_peak_val  = max(b_val_acc)
    r_peak_val  = max(r_val_acc)

    b_peak_epoch = b_val_acc.index(b_peak_val) + 1
    r_peak_epoch = r_val_acc.index(r_peak_val) + 1

    print("=" * 60)
    print("TASK 1 — QUANTITATIVE ANALYSIS SUMMARY")
    print("=" * 60)

    print("\n--- Train-Validation Set Performance ---")
    print("\n--- Baseline Model ---")
    print(f"  Final train accuracy      : {b_train_acc[-1]:.4f}")
    print(f"  Final validation accuracy : {b_val_acc[-1]:.4f}")
    print(f"  Generalisation gap        : {b_gap_final:.4f}")
    print(f"  Peak validation accuracy  : {b_peak_val:.4f} (epoch {b_peak_epoch})")

    print("\n--- Regularised Model ---")
    print(f"  Final train accuracy      : {r_train_acc[-1]:.4f}")
    print(f"  Final validation accuracy : {r_val_acc[-1]:.4f}")
    print(f"  Generalisation gap        : {r_gap_final:.4f}")
    print(f"  Peak validation accuracy  : {r_peak_val:.4f} (epoch {r_peak_epoch})")

    print("\n--- Gap Reduction ---")
    print(f"  Gap reduced by            : {b_gap_final - r_gap_final:.4f}")
    print(f"  Val accuracy improvement  : {r_peak_val - b_peak_val:.4f}")

    print("\n--- Confidence Calibration ---")
    print(f"  Baseline mean max confidence   : {base_conf:.4f}")
    print(f"  Regularised mean max confidence: {reg_conf:.4f}")
    print(f"  Reduction                      : {base_conf - reg_conf:.4f}")

    print("\n--- Test Set Performance ---")
    print("  Baseline Model:")
    print(f"    Test Loss     : {base_test_metrics['test_loss']:.4f}")
    print(f"    Test Accuracy : {base_test_metrics['test_accuracy']:.4f}")
    print("  Regularised Model:")
    print(f"    Test Loss     : {reg_test_metrics['test_loss']:.4f}")
    print(f"    Test Accuracy : {reg_test_metrics['test_accuracy']:.4f}")
    print("  Difference:")
    print(f"    Accuracy Gain : {reg_test_metrics['test_accuracy'] - base_test_metrics['test_accuracy']:.4f}")
    print(f"    Loss Change   : {base_test_metrics['test_loss'] - reg_test_metrics['test_loss']:.4f}")

    print("\n--- Technical Analysis ---")
    print("""
[REPLACE THIS BLOCK WITH YOUR ~500 WORD ANALYSIS COVERING:]
  1. The generalisation gap observed in each model
  2. Why the baseline overfits (high capacity, no regularisation)
  3. How SGD with momentum acts as implicit regularisation
  4. How dropout + weight decay shift the bias-variance position
  5. Justification of chosen hyperparameters
    """)
    print("=" * 60)


def main():
    # load histories
    b_history = load_history(MODEL_DIR / "baseline_train_history.json")
    r_history = load_history(MODEL_DIR / "regularised_train_history.json")

    b_epochs, b_train_acc, b_val_acc = extract_epoch_metrics(b_history)
    r_epochs, r_train_acc, r_val_acc = extract_epoch_metrics(r_history)

    # load models
    baseline_model    = load_model(dropout_prob=0.0, weights_path=MODEL_DIR / "baseline_model.pt")
    regularised_model = load_model(dropout_prob=0.5, weights_path=MODEL_DIR / "regularised_model.pt")
    print("Baseline model loaded:    ", type(baseline_model).__name__)
    print("Regularised model loaded: ", type(regularised_model).__name__)

    # test models
    _, test_dataset = download_data()
    base_batch_size = b_history["config"]["batch_size"]
    base_test_metrics, base_history_path = run_test_evaluation(
        baseline_model, test_dataset, base_batch_size,
        'baseline', MODEL_DIR, config=b_history['config'])
    reg_batch_size = r_history["config"]["batch_size"]
    reg_test_metrics, reg_history_path = run_test_evaluation(
        regularised_model, test_dataset, reg_batch_size,
        "regularised", MODEL_DIR, config=r_history["config"])

    # confidence calibration comparison
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=base_batch_size, shuffle=False)
    base_conf = evaluate_confidence(baseline_model, test_loader)
    reg_conf = evaluate_confidence(regularised_model, test_loader)

    # generate plot
    generate_gap_plot(b_epochs, b_train_acc, b_val_acc, r_epochs, r_train_acc,
                      r_val_acc, save_path=TASK_DIR / "generalisation_gap.png")

    # generate gap-per-epoch plot
    generate_gap_per_epoch_plot(b_epochs, b_train_acc, b_val_acc, r_epochs, r_train_acc,
                                r_val_acc, save_path=TASK_DIR / "gap_per_epoch.png")

    # compute summary values
    b_gap_final = b_train_acc[-1] - b_val_acc[-1]
    r_gap_final = r_train_acc[-1] - r_val_acc[-1]
    b_peak_val = max(b_val_acc)
    r_peak_val = max(r_val_acc)
    b_peak_epoch = b_val_acc.index(b_peak_val) + 1
    r_peak_epoch = r_val_acc.index(r_peak_val) + 1
    # build experiment summary
    summary = {
        "baseline": {
            "final_train_accuracy": b_train_acc[-1],
            "final_val_accuracy": b_val_acc[-1],
            "generalisation_gap": b_gap_final,
            "peak_val_accuracy": b_peak_val,
            "peak_val_epoch": b_peak_epoch,
            "test_metrics": base_test_metrics,
            "mean_confidence": base_conf
        },
        "regularised": {
            "final_train_accuracy": r_train_acc[-1],
            "final_val_accuracy": r_val_acc[-1],
            "generalisation_gap": r_gap_final,
            "peak_val_accuracy": r_peak_val,
            "peak_val_epoch": r_peak_epoch,
            "test_metrics": reg_test_metrics,
            "mean_confidence": reg_conf
        },
        "comparison": {
            "gap_reduction": b_gap_final - r_gap_final,
            "validation_accuracy_improvement": r_peak_val - b_peak_val,
            "confidence_reduction": base_conf - reg_conf,
            "test_accuracy_gain": reg_test_metrics["test_accuracy"] - base_test_metrics["test_accuracy"]
        }
    }
    save_json(summary, TASK_DIR / "task1_summary.json")
    # print analysis
    print_analysis(b_epochs, b_train_acc, b_val_acc, r_epochs, r_train_acc, r_val_acc, base_conf, reg_conf,
                   base_test_metrics, reg_test_metrics)
if __name__ == "__main__":
    main()