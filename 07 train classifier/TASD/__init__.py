from helper import format_seconds_to_time_string
from transformers import T5Tokenizer, T5ForConditionalGeneration
from TASD.preprocessing import CustomDatasetTASD, encode_example
from TASD.model import get_trainer_TASD
import numpy as np
import constants
import time


def train_TASD_model(LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset):
    results = {
        "LLM_NAME": LLM_NAME,
        "N_REAL": N_REAL,
        "N_SYNTH": N_SYNTH,
        "TARGET": TARGET,
        "LLM_SAMPLING": LLM_SAMPLING,
    }

    tokenizer = T5Tokenizer.from_pretrained(constants.MODEL_NAME_TASD)

    n_samples_train = []
    n_samples_test = []

    start_time = time.time()

    metrics_models = []

    for cross_idx in range(constants.N_FOLDS)[0:constants.TEST_FOLDS]:
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

        trainer = get_trainer_TASD(train_data, test_data, tokenizer, results, cross_idx)
        trainer.train()

        eval_metrics = trainer.evaluate()
        print(f"Split {cross_idx}:", eval_metrics)

        for metric in eval_metrics.keys():
            if metric not in results:
                results[metric] = []
                metrics_models.append(metric)

            results[metric].append(eval_metrics[metric])

    for metric in metrics_models:
        results[metric] = np.mean(results[metric])

    runtime = time.time() - start_time

    results["runtime"] = runtime
    results["runtime_formatted"] = format_seconds_to_time_string(runtime)
    results["n_samples_train"] = n_samples_train
    results["n_samples_train_mean"] = np.mean(n_samples_train)
    results["n_samples_test"] = n_samples_test
    results["n_samples_test_mean"] = np.mean(n_samples_test)

    return results
