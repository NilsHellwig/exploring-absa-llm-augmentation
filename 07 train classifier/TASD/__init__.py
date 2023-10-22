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

    start_time = time.time()

    metrics_models = []

    for cross_idx in range(constants.N_FOLDS)[0:constants.TEST_FOLDS]:
        # Load Data
        train_data = train_dataset[cross_idx]
        test_data = test_dataset[cross_idx]
        train_data = CustomDatasetTASD([encode_example(example, tokenizer)["input_ids"] for example in train_data],
                                       [encode_example(example, tokenizer)["attention_mask"]
                                           for example in train_data],
                                       [encode_example(example, tokenizer)["labels"] for example in train_data])
        test_data = CustomDatasetTASD([encode_example(example, tokenizer)["input_ids"] for example in test_data],
                                      [encode_example(example, tokenizer)["attention_mask"]
                                       for example in test_data],
                                      [encode_example(example, tokenizer)["labels"] for example in test_data])

        trainer = get_trainer_TASD(train_data, test_data, tokenizer)
        trainer.train()

        eval_metrics = trainer.evaluate()

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

    return results
