from transformers import AutoTokenizer
from helper import format_seconds_to_time_string
from transformers import DataCollatorWithPadding
from ACD.preprocessing import preprocess_data_ACD
from ACD.model import create_model_ACD, get_trainer_ACD
from transformers import AutoTokenizer
import numpy as np
import constants
import time


def train_ACD_model(LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset):
    results = {
        "LLM_NAME": LLM_NAME,
        "N_REAL": N_REAL,
        "N_SYNTH": N_SYNTH,
        "TARGET": TARGET,
        "LLM_SAMPLING": LLM_SAMPLING,
    }

    loss = []

    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME_ACD)

    metrics_prefixes = ["accuracy", "hamming_loss", "f1_macro", "f1_micro", "f1_weighted"] + [f"{m}_{ac}" for ac in constants.ASPECT_CATEGORIES for m in ["precision", "recall", "f1"]]
    metrics_total = {f"{m}": [] for m in metrics_prefixes}
    

    for cross_idx in range(constants.N_FOLDS)[0:constants.TEST_FOLDS]:
        # Load Data
        train_data = preprocess_data_ACD(train_dataset[cross_idx], tokenizer)
        test_data = preprocess_data_ACD(test_dataset[cross_idx], tokenizer)


        # Train Model
        trainer = get_trainer_ACD(train_data, test_data, tokenizer)
        trainer.train()

        # Save Evaluation of Test Data
        eval_metrics = trainer.evaluate()

        # Save Metrics for fold
        for m in metrics_prefixes:
            metrics_total[f"{m}"] = eval_metrics[f"eval_{m}"]

        loss.append(eval_metrics["eval_loss"])

    runtime = time.time() - start_time

    results["eval_loss"] = np.mean(loss)

    results.update({f"eval_{m}": np.mean(metrics_total[f"{m}"]) for m in metrics_prefixes})

    results["runtime"] = runtime
    results["runtime_formatted"] = format_seconds_to_time_string(runtime)
    return results
