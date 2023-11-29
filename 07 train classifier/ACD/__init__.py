from transformers import AutoTokenizer
from helper import format_seconds_to_time_string
from ACD.preprocessing import preprocess_data_ACD
from ACD.model import get_trainer_ACD
from transformers import AutoTokenizer
import numpy as np
import constants
import time


def train_ACD_model(LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset, validation_dataset):
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
    n_samples_validation = []
    n_epochs_best_validation_score = []
    log_history = {}

    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME_ACD)

    metrics_prefixes = ["accuracy", "hamming_loss", "f1_macro", "f1_micro", "f1_weighted"] + [
        f"{m}_{ac}" for ac in constants.ASPECT_CATEGORIES for m in ["precision", "recall", "f1", "accuracy"]]
    metrics_total = {f"{m}": [] for m in metrics_prefixes}

    for cross_idx in range(constants.N_FOLDS)[0:constants.TEST_FOLDS]:
        # Load Data
        train_data = preprocess_data_ACD(train_dataset[cross_idx], tokenizer)
        test_data = preprocess_data_ACD(test_dataset[cross_idx], tokenizer)
        validation_data = preprocess_data_ACD(validation_dataset[cross_idx], tokenizer)

        n_samples_train.append(len(train_data))
        n_samples_test.append(len(test_data))
        n_samples_validation.append(len(validation_data))

        # Train Model
        trainer = get_trainer_ACD(train_data, validation_data, tokenizer, results, cross_idx)
        trainer.train()

        # Get index of best epoch on validation data / save log history
        n_epochs_best_validation_score.append(trainer.state.epoch - constants.N_EPOCHS_EARLY_STOPPING_PATIENCE)
        log_history[cross_idx] = trainer.state.log_history

        # Save Evaluation of Test Data
        eval_metrics = trainer.evaluate(test_data)
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
    results["n_samples_validation"] = n_samples_validation
    results["n_samples_validation_mean"] = np.mean(n_samples_validation)
    results["n_epochs_best_validation_score"] = n_epochs_best_validation_score
    results["n_epochs_best_validation_score_mean"] = np.mean(n_epochs_best_validation_score)
    results["log_history"] = log_history

    return results
