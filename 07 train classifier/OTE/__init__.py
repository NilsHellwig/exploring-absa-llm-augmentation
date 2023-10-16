from helper import format_seconds_to_time_string
from OTE.preprocessing import get_preprocessed_data_OTE
from OTE.model import get_trainer_OTE
from transformers import AutoTokenizer
import numpy as np
import constants
import time


def train_OTE_model(LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset):
    results = {
        "LLM_NAME": LLM_NAME,
        "N_REAL": N_REAL,
        "N_SYNTH": N_SYNTH,
        "TARGET": TARGET,
        "LLM_SAMPLING": LLM_SAMPLING,
    }

    f1_micro_scores = []
    eval_loss = []

    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME_OTE)

    start_time = time.time()

    for cross_idx in range(constants.N_FOLDS)[0:constants.TEST_FOLDS]:
        # Load Data
        train_data = train_dataset[cross_idx]
        test_data = test_dataset[cross_idx]
        train_data, test_data = get_preprocessed_data_OTE(
            train_data, test_data, tokenizer)
        trainer = get_trainer_OTE(train_data, test_data, tokenizer)
        trainer.train()

        # Save Evaluation Metrics
        eval_metrics = trainer.evaluate()
        f1_micro_scores.append(eval_metrics["eval_f1_micro"])
        eval_loss.append(eval_metrics["eval_loss"])

    runtime = time.time() - start_time
    results["runtime"] = runtime
    results["runtime_formatted"] = format_seconds_to_time_string(runtime)
    results["eval_loss"] = np.mean(eval_loss)
    results["f1_micro"] = np.mean(f1_micro_scores)
    return results
