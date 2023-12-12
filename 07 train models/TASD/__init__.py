from helper import format_seconds_to_time_string
from transformers import T5Tokenizer, T5ForConditionalGeneration
from TASD.preprocessing import CustomDatasetTASD, encode_example
from TASD.model import get_trainer_TASD
import numpy as np
import constants
import time


def train_TASD_model(LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset, validation_dataset):
    results_meta = {
        "LLM_NAME": LLM_NAME,
        "N_REAL": N_REAL,
        "N_SYNTH": N_SYNTH,
        "TARGET": TARGET,
        "LLM_SAMPLING": LLM_SAMPLING,
    }

    results = results_meta.copy()

    loss = []
    n_samples_train = []
    n_samples_test = []
    metrics_models = []
    n_samples_validation = []
    n_epochs_best_validation_score = []
    log_history = {}
    
    start_time = time.time()

    tokenizer = T5Tokenizer.from_pretrained(constants.MODEL_NAME_TASD)

    for cross_idx in range(constants.N_FOLDS):
        # Load Data
        train_data = train_dataset[cross_idx]
        test_data = test_dataset[cross_idx]
        valid_data = validation_dataset[cross_idx]

        n_samples_train.append(len(train_data))
        n_samples_test.append(len(test_data))
        n_samples_validation.append(len(valid_data))

        train_data = CustomDatasetTASD([encode_example(example, tokenizer)["input_ids"] for example in train_data],
                                       [encode_example(example, tokenizer)["attention_mask"]
                                           for example in train_data],
                                       [encode_example(example, tokenizer)["labels"] for example in train_data])
        test_data = CustomDatasetTASD([encode_example(example, tokenizer)["input_ids"] for example in test_data],
                                      [encode_example(example, tokenizer)["attention_mask"]
                                       for example in test_data],
                                      [encode_example(example, tokenizer)["labels"] for example in test_data])
        valid_data = CustomDatasetTASD([encode_example(example, tokenizer)["input_ids"] for example in valid_data],
                                      [encode_example(example, tokenizer)["attention_mask"]
                                       for example in valid_data],
                                      [encode_example(example, tokenizer)["labels"] for example in valid_data])
        
        # Train Model
        trainer = get_trainer_TASD(train_data, valid_data, tokenizer, results_meta, cross_idx)
        trainer.train()

        # Get index of best epoch on validation data / save log history
        n_epochs_best_validation_score.append(trainer.state.epoch - constants.N_EPOCHS_EARLY_STOPPING_PATIENCE)
        log_history[cross_idx] = trainer.state.log_history

        # Save Evaluation of Test Data
        eval_metrics = trainer.evaluate(test_data)
        print(f"Split {cross_idx}:", eval_metrics)

        # Save Metrics for fold
        for metric in eval_metrics.keys():
            if metric not in results:
                results[metric] = []
                metrics_models.append(metric)

            results[metric].append(eval_metrics[metric])
        
        loss.append(eval_metrics["eval_loss"])
    
    runtime = time.time() - start_time

    results["eval_loss"] = np.mean(loss)
    
    for metric in metrics_models:
        results[metric] = np.mean(results[metric])


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
