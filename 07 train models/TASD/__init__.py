from helper import format_seconds_to_time_string
from transformers import T5Tokenizer
from TASD.preprocessing import CustomDatasetTASD, encode_example
from TASD.model import get_trainer_TASD
import subprocess
import numpy as np
import constants
import shutil
import time


def train_TASD_model(LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset):
    results_meta = {
        "LLM_NAME": LLM_NAME,
        "N_REAL": N_REAL,
        "N_SYNTH": N_SYNTH,
        "TARGET": TARGET,
        "LLM_SAMPLING": LLM_SAMPLING,
        "single_split_results": []
    }

    results = results_meta.copy()

    loss = []
    n_samples_train = []
    n_samples_test = []
    metrics_models = []
    log_history = {}
    
    start_time = time.time()

    tokenizer = T5Tokenizer.from_pretrained(constants.MODEL_NAME_TASD)

    for cross_idx in range(constants.N_FOLDS):
        # Load Data
        train_data = train_dataset[cross_idx]
        test_data = test_dataset[cross_idx]

        n_samples_train.append(len(train_data))
        n_samples_test.append(len(test_data))

        train_data = CustomDatasetTASD([encode_example(example, tokenizer)["input_ids"] for example in train_data],
                                       [encode_example(example, tokenizer)["attention_mask"]
                                           for example in train_data],
                                       [encode_example(example, tokenizer)["labels"] for example in train_data])
        test_data = CustomDatasetTASD([encode_example(example, tokenizer)["input_ids"] for example in test_data],
                                      [encode_example(example, tokenizer)["attention_mask"]
                                       for example in test_data],
                                      [encode_example(example, tokenizer)["labels"] for example in test_data])
        
        # Train Model
        trainer = get_trainer_TASD(train_data, test_data, tokenizer, results_meta, cross_idx)
        trainer.train()

        # save log history
        log_history[cross_idx] = trainer.state.log_history

        # Save Evaluation of Test Data
        eval_metrics = trainer.evaluate(test_data)
        print(f"Split {cross_idx}:", eval_metrics)

         # Save Evaluation of Split
        results["single_split_results"].append(eval_metrics)


        # Save Metrics for fold
        for metric in eval_metrics.keys():
            if metric not in results:
                results[metric] = []
                metrics_models.append(metric)

            results[metric].append(eval_metrics[metric])
        
        loss.append(eval_metrics["eval_loss"])

        path_output = constants.OUTPUT_DIR_TASD + "_" + results["LLM_NAME"]+"_"+str(results["N_REAL"])+"_"+str(results["N_SYNTH"]) + "_"+results["TARGET"]+"_"+results["LLM_SAMPLING"]+"_"+str(cross_idx)
        shutil.rmtree(path_output)

        subprocess.call("rm -rf /home/mi/.local/share/Trash", shell=True)
    
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
    results["log_history"] = log_history

    return results
