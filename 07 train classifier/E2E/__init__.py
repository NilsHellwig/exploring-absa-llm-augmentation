from E2E.preprocessing import get_preprocessed_data_E2E
from helper import format_seconds_to_time_string
from E2E.model import get_trainer_E2E
from transformers import AutoTokenizer
import numpy as np
import constants
import time


def train_E2E_model(LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset):
    results = {
        "LLM_NAME": LLM_NAME,
        "N_REAL": N_REAL,
        "N_SYNTH": N_SYNTH,
        "TARGET": TARGET,
        "LLM_SAMPLING": LLM_SAMPLING,
    }

    metrics_classwise_prefix = [
        "accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"]
    metrics_total_prefix = ["f1_micro", "f1_macro", "precision",
                            "recall", "accuracy", "f1", "tp", "tn", "fp", "fn"]
    metrics_total = {f"{metric}": [] for metric in metrics_total_prefix}
    metrics_total.update({f"{metric}_{ac}": []
                         for metric in metrics_classwise_prefix for ac in constants.POLARITIES})

    eval_loss = []
    n_samples_train = []
    n_samples_test = []

    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME_E2E)

    start_time = time.time()

    for cross_idx in range(constants.N_FOLDS)[0:constants.TEST_FOLDS]:
        # Load Data
        train_data = train_dataset[cross_idx]
        test_data = test_dataset[cross_idx]

        train_data, test_data = get_preprocessed_data_E2E(
            train_data, test_data, tokenizer)
        trainer = get_trainer_E2E(train_data, test_data, tokenizer)

        trainer.train()

        eval_metrics = trainer.evaluate()
        print(f"Split {cross_idx}:", eval_metrics)
        eval_loss.append(eval_metrics["eval_loss"])
        n_samples_train.append(len(train_data))
        n_samples_test.append(len(test_data))

        for m in metrics_total_prefix:
            metrics_total[m].append(eval_metrics[f"eval_{m}"])

        for polarity in constants.POLARITIES:
            for classwise_metric in metrics_classwise_prefix:
                metrics_total[f"{classwise_metric}_{polarity}"].append(
                    eval_metrics[f"eval_{classwise_metric}_{polarity}"])

    runtime = time.time() - start_time

    results.update({f"eval_{m}": np.mean(metrics_total[f"{m}"]) for m in metrics_total_prefix})
    results.update({f"eval_{m}_{polarity}": np.mean(metrics_total[f"{m}_{polarity}"]) for m in metrics_classwise_prefix for polarity in constants.POLARITIES})

    results["runtime"] = runtime
    results["runtime_formatted"] = format_seconds_to_time_string(runtime)
    results["eval_loss"] = np.mean(eval_loss)
    results["n_samples_train"] = n_samples_train
    results["n_samples_train_mean"] = np.mean(n_samples_train)
    results["n_samples_test"] = n_samples_test
    results["n_samples_test_mean"] = np.mean(n_samples_test)

    return results