# GenAI usage statement: Claude (Anthropic) was used in an assistive role to help
# structure and debug this file. All deep learning logic, analysis, and design
# decisions are the author's own.

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from utils.plotting import generate_gap_plot, generate_gap_per_epoch_plot
from utils.common import *
from utils.experiment import *


TASK_DIR   = Path(__file__).resolve().parent
BASE_DIR = TASK_DIR / "models" / "baseline"
REG_DIR = TASK_DIR / "models" / "regularised"
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_analysis(b_epochs, b_train_acc, b_val_acc, r_epochs, r_train_acc, r_val_acc, base_conf,
                   reg_conf, base_test_metrics, reg_test_metrics):
    """
    Prints a detailed quantitative and qualitative analysis comparing
    baseline and regularised models, including generalisation gap,
    calibration, and test performance.

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
Task 1: Generalisation Gap and Regularisation
This experiment trained a relatively high capacity CNN on CIFAR-10 to investigate the bias variance trade off and generalisation gap. The model uses convolutional layers (3, 32, 64 channels) followed by residual blocks at 64, 128, and 256 channels with BatchNorm, SiLU, and Squeeze and Excitation attention. Spatial resolution is reduced via max pooling, and classification uses global average pooling with a linear output layer. Two regimes were compared, a baseline with minimal explicit regularisation and a regularised model using augmentation, dropout, and weight decay, with hyperparameters obtained via successive halving.

The baseline model clearly overfits. Training accuracy reached 100% while validation accuracy plateaued at 88.9% ( generalisation gap 11.2%). Training loss approached zero (0.0000298), whereas validation loss reached a minimum of 0.4726 at epoch 8 and increased afterwards reaching 0.643 by epoch 80, indicating memorisation rather than generalisable feature learning. This demonstrates high variance, the model has eough capacity (millions of parameters relative to the 50k CIFAR-10 training samples) to fit every training sample exactly but this does not transfer to unseen data. High confidence predictions (96.7%) relative to test accuracy (87.9%) further indicate overconfidence and poor calibration, this is likely caused by the optimiser pushing logits toward extreme values to minimise the already very small training loss.

Even though the baseline was overfitting it still achieves 87.9% test accuracy. This is because SGD with momentum acts as implicit regularisation. Each gradient update is estimated from a batch of 64 samples which is only 0.16% of the training set. This introduces gradient noise which is believed to bias optimisation toward flatter minimums on the loss surface. Flat minimums generalise better than sharp minimums because they are less precisely tuned to the specific training examples, this means that small changes between the train and test data do not cause large loss increases. Across 80 epochs the model does 31250 gradient updates, and so SGDs radnomness accumulates and we observe a meaningful implicit regularisation effect even with no explicit regularisers. The momentum value of 0.929 selected by the hyperparameter search is relatively high, this smooths updates across past gradients reducing sensitivity to individual noisy batches and supporting stable convergence.
SGD's implicit regularisation alone was not enough to close the gap. The regularised model added three regularisation techniques ontop of the baseline. These were random horizontal flip, random crop, and cutout augmentation applied to training data only, dropout at 21.9% on the FC classifier layer, and weight decay at 1.2e-6. These methods reduce variance by limiting the model’s reliance on specific training examples. Augmentation exposes the model to input variations, dropout prevents coadaptation by forcing redundant representations, and weight decay discourages overly large highly specific parameter values leading to more robust solutions.  The hyperparameter search suggested that a dropout value of 0.219 balanced bias and variance effectively for this architecture,  higher values tended to causes over regularisation given that BatchNorm, SE attention, and augmentation are already providing implicit regularisation.
The regularised model substantially reduced the generalisation gap (3.0% vs 11.2%), achieving 92.6% validation accuracy and 91.3% test accuracy, with test loss decreasing from 0.728 to 0.332. Validation loss improved until epoch 57 unlike the baseline which diverged early demonstrating the model carried on learning throughout training. Overall, regularisation reduced variance with only a small increase in bias evidenced by a 4.4% drop in training accuracy.

I initially applied Cutout before normalisation but corrected this following GenAI feedback by applying it after normalisation to ensure consistent feature removal.
    """)
    print("=" * 60)


def main():
    """
        Runs Task 1 evaluation pipeline.

        This function:
        - Loads training histories for baseline and regularised models
        - Extracts training and validation metrics
        - Loads trained model weights
        - Evaluates both models on the test set
        - Computes confidence calibration metrics
        - Generates generalisation gap plots
        - Saves a summary of results to JSON
        - Prints a detailed technical analysis

        Outputs:
            - generalisation_gap.png
            - gap_per_epoch.png
            - task1_summary.json
            - Printed analysis to terminal
    """
    # load histories
    b_history = load_history(BASE_DIR / "baseline_train_history.json")
    r_history = load_history(REG_DIR / "regularised_train_history.json")

    b_epochs, b_train_acc, b_val_acc = extract_epoch_metrics(b_history)
    r_epochs, r_train_acc, r_val_acc = extract_epoch_metrics(r_history)

    # load models
    baseline_model    = load_model(dropout_prob=b_history["config"].get("reg_dropout", 0.0),
                                   weights_path=BASE_DIR / "baseline_model.pt")
    regularised_model = load_model(dropout_prob=r_history["config"].get("reg_dropout", 0.0),
                                   weights_path=REG_DIR / "regularised_model.pt")
    print("Baseline model loaded:    ", type(baseline_model).__name__)
    print("Regularised model loaded: ", type(regularised_model).__name__)

    # test models
    _, test_dataset = download_data()
    base_batch_size = b_history["config"]["batch_size"]
    base_test_metrics, base_history_path = run_test_evaluation(
        baseline_model, test_dataset, base_batch_size,
        'baseline', BASE_DIR, config=b_history['config'])
    reg_batch_size = r_history["config"]["batch_size"]
    reg_test_metrics, reg_history_path = run_test_evaluation(
        regularised_model, test_dataset, reg_batch_size,
        "regularised", REG_DIR, config=r_history["config"])

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