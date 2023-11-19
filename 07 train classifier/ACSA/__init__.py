from ACSA.preprocessing import preprocess_data_ACSA
from helper import format_seconds_to_time_string
from ACSA.model import get_trainer_ACSA
from transformers import AutoTokenizer
import numpy as np
import constants
import time


def train_ACSA_model(LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset):
    results = {
        "LLM_NAME": LLM_NAME,
        "N_REAL": N_REAL,
        "N_SYNTH": N_SYNTH,
        "TARGET": TARGET,
        "LLM_SAMPLING": LLM_SAMPLING,
    }

    loss = []
    n_samples_train = []
    n_samples_test = []

    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME_ACSA)
    metrics_prefixes = ["accuracy", "hamming_loss", "f1_macro", "f1_micro", "f1_weighted"] + [
        f"{m}_{ac}" for ac in constants.ASPECT_CATEGORIES for m in ["precision", "recall", "f1", "accuracy"]] + [
        f"{m}_{ac}-{polarity}" for ac in constants.ASPECT_CATEGORIES for polarity in constants.POLARITIES for m in ["precision", "recall", "f1", "accuracy"]]
    metrics_total = {f"{m}": [] for m in metrics_prefixes}

    for cross_idx in range(constants.N_FOLDS)[0:constants.TEST_FOLDS]:
        # Load Data
        train_data = preprocess_data_ACSA(train_dataset[cross_idx], tokenizer)
        test_data = preprocess_data_ACSA(test_dataset[cross_idx], tokenizer)

        n_samples_train.append(len(train_data))
        n_samples_test.append(len(test_data))

        # Train Model
        trainer = get_trainer_ACSA(train_data, test_data, tokenizer, results, cross_idx)
        trainer.train()

        # Save Evaluation of Test Data
        eval_metrics = trainer.evaluate()
        print(f"Split {cross_idx}:", eval_metrics)

        # Save Metrics for fold
        for m in metrics_prefixes:
            metrics_total[f"{m}"] = eval_metrics[f"eval_{m}"]

        loss.append(eval_metrics["eval_loss"])

    runtime = time.time() - start_time

    results["eval_loss"] = np.mean(loss)

    results.update({f"eval_{m}": np.mean(
        metrics_total[f"{m}"]) for m in metrics_prefixes})

    results["runtime"] = runtime
    results["runtime_formatted"] = format_seconds_to_time_string(runtime)
    results["n_samples_train"] = n_samples_train
    results["n_samples_train_mean"] = np.mean(n_samples_train)
    results["n_samples_test"] = n_samples_test
    results["n_samples_test_mean"] = np.mean(n_samples_test)

    return results
