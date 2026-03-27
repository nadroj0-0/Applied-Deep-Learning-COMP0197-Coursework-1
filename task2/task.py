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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_analysis(history, noisy_test_metrics, noise_results, exp_name):
    """
    Print quantitative summary statistics

    Args:
        history (dict): Training history loaded from JSON.
        noisy_test_metrics (dict): Metrics from evaluation on the noisy test set.
        noise_results (dict): Accuracy results across multiple noise levels.
    """
    metrics = history["metrics"]["epoch_metrics"]
    train_acc = [m.get("train_accuracy") for m in metrics]
    val_acc = [m["validation_accuracy"] for m in metrics]

    best_val_acc = max(val_acc) if val_acc else None
    best_val_epoch = val_acc.index(best_val_acc) + 1 if val_acc else None
    final_train_acc = train_acc[-1] if train_acc else None
    final_val_acc = val_acc[-1] if val_acc else None

    print("=" * 60)
    print(f"{exp_name} — QUANTITATIVE ANALYSIS SUMMARY")
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

    print("=" * 60)

def tech_analysis():
    print("\n--- Technical Analysis ---")
    print("""Task 2: MixUp and Label Smoothing"
          This experiment extended on Task 1 by incorporating MixUp augmentation and label smoothing to investigate their effect on model robustness and generalisation. The primary model of interest is the regularised one, which inherits the hyperparameters from Task 1 and adds MixUp and label smoothing on top (found via search). A baseline model with identical MixUp and label smoothing parameters but without regularisation was also trained providing a controlled comparison to isolate the effect of regularisation.
MixUp prevents memorisation by training the network on convex combinations of input pairs rather than individual examples. Given two training samples a mixing coefficient λ~Beta(α,α) is sampled and the blended input x̃ = λx_i + (1−λ)x_j is presented to the network alongside a similarly blended soft target. Because the model never sees any training image in isolation it cannot learn to associate exact pixel patterns with hard labels. Instead it is forced to learn linear interpolations between classes which encourages smoother decision boundarie. Representations become less sample specific and more structured,  the network learns the geometry of the class manifold rather than individual sample locations within it. The selected α=0.119 is conservative giving only mild mixing where λ stays close to 0 or 1 most of the time, which is appropriate given that data augmentation is already providing variance in the training distribution.
Label smoothing addresses a complementary failure mode. Standard cross entropy loss with hard targets (1.0 for the correct class, 0.0 for all others) means the model can always reduce loss further by pushing logits toward + or - infinity. This causes the optimiser to overshoot producing overconfident predictions that do not reflect genuine uncertainty. Label smoothing replaces the hard target with a soft distribution, the correct class receives probability 1−ε and the remaining probability ε is spread uniformly across all other classes. In this implementation ε=0.048, meaning the correct class target is 0.952 and each incorrect class receives 0.005. This creates a finite optimisation target, and helps prevent logits to from growing large and producing better confidence estimates.
Early stopping with patience=5 was implemented by tracking the validation loss, saving the best model state at each improvement and restoring the best one. The regularised model peaked at epoch 29 with 91.0% validation accuracy and early stopping triggered at epoch 34 recovering the epoch 29 weights. The baseline model peaked at epoch 20 and stopped at epoch 25.
The noisy test evaluation adds Gaussian noise (σ=0.1) to CIFAR-10 test images. The regularised model achieves 62.9% accuracy on this noisy set versus 66.3% for the baseline,  this shows sliht differences in how the models shape decision boundaries under noisy inputs. A robustness curve was also generated so see how the models perform with varying levels of noise. Both models show a sharp degradation beyond moderate noise levels, indicating that while MixUp and label smoothing improve clean generalisation they do not ensure robustness to large noise. The regularised model achieves higher clean test accuracy (90.5% vs 87.6%) but slightly lower accuracy under noise (62.9% vs 66.3%) suggesting a trade off between generalisation and robustness.
GenAI suggested evaluating robustness across multiple noise levels instead of a single value providing a clearer view of performance under increasing noise.
""")

def evaluate_noisy_test(model, test_dataset, batch_size, name, config, exp_dir):
    """
    Evaluates a trained model on a noisy version of the test dataset.

    Gaussian noise is added to test inputs to assess robustness.

    Args:
        model (torch.nn.Module): Trained model.
        test_dataset (Dataset): CIFAR-10 test dataset.
        batch_size (int): Batch size for evaluation.
        name (str): Experiment name.
        config (dict): Training configuration.
        exp_dir (Path): Directory for saving outputs.

    Returns:
        tuple:
            - dict: Test loss and accuracy on noisy data.
            - Path: Path to saved evaluation history.
    """
    test_loader = build_noisy_test_loader(test_dataset, batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(test_loader, model, criterion)
    print("\nNoisy test performance")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    test_metrics = {"test_loss": test_loss,"test_accuracy": test_acc}
    history_path = save_history(test_metrics,name,"noisy_test",model,exp_dir,config=config)
    return test_metrics, history_path



def main():
    """
        Runs the Task 2 evaluation pipeline.

        This function:
        - Loads trained models and their histories
        - Evaluates performance on noisy test data
        - Computes robustness across multiple noise levels
        - Generates MixUp visualisation examples
        - Prints quantitative summaries for each experiment
        - Saves a consolidated results summary to JSON
        - Outputs a technical analysis of findings

        Outputs:
            - noise_robustness.json
            - robustness_demo.png
            - task2_summary.json
            - Printed analysis to terminal
    """
    EXPERIMENTS = {
        "baseline_fixed_mixup_ls": {"dropout_prob": 0.0},
        #"baseline_free_mixup_ls": {"dropout_prob": 0.0},   # additional model, uncomment if you want to run
        "regularised_fixed_mixup_ls": {"dropout_prob": None},  # read from config
        #"regularised_free_mixup_ls": {"dropout_prob": None},  # additional model, uncomment if you want to run
    }
    # Load CIFAR10
    _, test_dataset = download_data()
    summary = {}
    for exp_name, opts in EXPERIMENTS.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {exp_name}")
        print(f"{'=' * 60}")
        exp_dir = TASK_DIR / "models" / exp_name
        # Load training history
        history = load_history(exp_dir / f"{exp_name}_train_history.json")
        config = history["config"]
        batch_size = config["batch_size"]
        # Load trained model
        dropout = opts["dropout_prob"] if opts["dropout_prob"] is not None else config.get("reg_dropout", 0.0)
        model = load_model(dropout_prob=dropout, weights_path=exp_dir / f"{exp_name}_model.pt")
        print("Model loaded:", type(model).__name__)
        # Evaluate on noisy test set
        noisy_test_metrics, _ = evaluate_noisy_test(model,test_dataset,batch_size,exp_name,config, exp_dir)
        # noise robustness curve
        noise_results = evaluate_noise_robustness(model, test_dataset, batch_size, exp_dir / "noise_robustness.json")
        # Generate MixUp demo figure
        save_mixup_demo(mixup_data,test_dataset, exp_dir / "robustness_demo.png",alpha=config.get("mixup_alpha", 0.4),
                        device=device)
        # Print analysis summary
        print_analysis(history, noisy_test_metrics, noise_results, exp_name)
        metrics = history["metrics"]["epoch_metrics"]
        val_acc = [m["validation_accuracy"] for m in metrics]
        summary[exp_name] = {
            "config": config,
            "final_val_accuracy": val_acc[-1],
            "peak_val_accuracy": max(val_acc),
            "noisy_test_metrics": noisy_test_metrics,
            "noise_robustness_curve": noise_results
        }
    save_json(summary, TASK_DIR / "task2_summary.json")
    tech_analysis()

if __name__ == "__main__":
    main()
