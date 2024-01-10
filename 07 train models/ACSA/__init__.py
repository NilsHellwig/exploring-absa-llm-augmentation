from ACSA.preprocessing import preprocess_data_ACSA
from helper import format_seconds_to_time_string
from ACSA.model import get_trainer_ACSA
from transformers import AutoTokenizer
import subprocess
import numpy as np
import constants
import time
import shutil


def train_ACSA_model(LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset, validation_dataset):
    results = {
        "LLM_NAME": LLM_NAME,
        "N_REAL": N_REAL,
        "N_SYNTH": N_SYNTH,
        "TARGET": TARGET,
        "LLM_SAMPLING": LLM_SAMPLING,
        "single_split_results": []
    }

    loss = []
    n_samples_train = []
    n_samples_test = []
    n_samples_validation = []
    n_epochs_best_validation_score = []
    log_history = {}

    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME_ACSA)
    metrics_prefixes = ["accuracy", "hamming_loss", "f1_macro", "f1_micro", "f1_weighted"] + [
        f"{m}_{ac}" for ac in constants.ASPECT_CATEGORIES for m in ["precision", "recall", "f1", "accuracy", "n_examples"]] + [
        f"{m}_{ac}-{polarity}" for ac in constants.ASPECT_CATEGORIES for polarity in constants.POLARITIES for m in ["precision", "recall", "f1", "accuracy", "n_examples"]]
    metrics_total = {f"{m}": [] for m in metrics_prefixes}

    for cross_idx in range(constants.N_FOLDS):
        # Load Data
        train_data = preprocess_data_ACSA(train_dataset[cross_idx], tokenizer)
        test_data = preprocess_data_ACSA(test_dataset[cross_idx], tokenizer)
        validation_data = preprocess_data_ACSA(validation_dataset[cross_idx], tokenizer)

        n_samples_train.append(len(train_data))
        n_samples_test.append(len(test_data))
        n_samples_validation.append(len(validation_data))

        # Train Model
        trainer = get_trainer_ACSA(train_data, validation_data, tokenizer, results, cross_idx)
        trainer.train()

        # Get index of best epoch on validation data / save log history
        n_epochs_best_validation_score.append(trainer.state.epoch - constants.N_EPOCHS_EARLY_STOPPING_PATIENCE)
        log_history[cross_idx] = trainer.state.log_history

        # Save Evaluation of Test Data
        eval_metrics = trainer.evaluate(test_data)
        print(f"Split {cross_idx}:", eval_metrics)

        # Save Evaluation of Split
        results["single_split_results"].append(eval_metrics)

        # Save Metrics for fold
        for m in metrics_prefixes:
            metrics_total[f"{m}"].append(eval_metrics[f"eval_{m}"])

        loss.append(eval_metrics["eval_loss"])

        path_output = constants.OUTPUT_DIR_ACSA + "_" + results["LLM_NAME"]+"_"+str(results["N_REAL"])+"_"+str(results["N_SYNTH"]) + "_"+results["TARGET"]+"_"+results["LLM_SAMPLING"]+"_"+str(cross_idx)
        shutil.rmtree(path_output)

        subprocess.call("rm -rf /home/mi/.local/share/Trash", shell=True)

    runtime = time.time() - start_time

    results["eval_loss"] = np.mean(loss)

    results.update({f"eval_{m}": np.mean(
        metrics_total[f"{m}"]) for m in metrics_prefixes})
    
    # score correction for ac + pol with less than 1 example
    for ac in constants.ASPECT_CATEGORIES:
        for polarity in constants.POLARITIES:
            n_examples_category_polarity = metrics_total[f"n_examples_{ac}-{polarity}"]
            for metric in ["precision", "recall", "f1", "accuracy"]:
                # in dieser liste speichere ich die berechnete metrik (maximal 5 zahlen in der liste), 
                # sofern es in dem durchgang mind 1 beispiel gab, für die gegebene kategorie + polarität
                calculated_metrics_more_1_samples = []
                single_metric_results = metrics_total[f"{metric}_{ac}-{polarity}"]
                for i in range(constants.N_FOLDS):
                    if n_examples_category_polarity[i] > 0:
                        calculated_metrics_more_1_samples.append(single_metric_results[i])
                results.update({f"eval_{metric}_{ac}-{polarity}": np.mean(calculated_metrics_more_1_samples)})
            results.update({f"eval_n_folds_for_metric_calculation_{ac}-{polarity}": sum([1 if n>0 else 0 for n in n_examples_category_polarity])})


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
