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

    f1_micro_scores = []
    f1_macro_scores = []
    f1_weighted_scores = []
    accuracy_scores = []
    class_f1_scores = []
    loss = []
    hamming = []

    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    for cross_idx in range(constants.N_FOLDS)[0:constants.TEST_FOLDS]:
        # Load Data
        train_data = preprocess_data_ACD(train_dataset[cross_idx], tokenizer)
        test_data = preprocess_data_ACD(test_dataset[cross_idx], tokenizer)

        # Load Model
        model_ACD = create_model_ACD()

        # Train Model
        trainer = get_trainer_ACD(model_ACD, train_data, test_data, tokenizer)
        trainer.train()

        # Save Evaluation of Test Data
        eval_metrics = trainer.evaluate()

        # Save Metrics for fold
        f1_micro_scores.append(eval_metrics["eval_f1_micro"])
        f1_macro_scores.append(eval_metrics["eval_f1_macro"])
        f1_weighted_scores.append(eval_metrics["eval_f1_weighted"])
        accuracy_scores.append(eval_metrics["eval_accuracy"])
        class_f1_scores.append(eval_metrics["eval_class_f1_scores"])
        loss.append(eval_metrics["eval_loss"])
        hamming.append(eval_metrics["eval_hamming_loss"])

    runtime = time.time() - start_time

    results["loss"] = np.mean(loss)
    results["hamming"] = np.mean(hamming)
    results["accuracy"] = np.mean(accuracy_scores)
    results["f1_micro"] = np.mean(f1_micro_scores)
    results["f1_macro"] = np.mean(f1_macro_scores)
    results["f1_weighted"] = np.mean(f1_weighted_scores)
    results["runtime"] = runtime
    results["runtime_formatted"] = format_seconds_to_time_string(runtime)
    return results
