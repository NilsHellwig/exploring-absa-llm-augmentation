from ACSA.model import get_trainer_ACSA
from ACSA.preprocessing import preprocess_data_ACSA
from transformers import AutoTokenizer
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
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME_ACSA)
    metrics_prefixes = ["accuracy", "hamming_loss", "f1_macro", "f1_micro", "f1_weighted"] + [
        f"{m}_{ac}" for ac in constants.ASPECT_CATEGORIES for m in ["precision", "recall", "f1", "accuracy"]] + [
        f"{m}_{ac}_{polarity}" for ac in constants.ASPECT_CATEGORIES for polarity in constants.POLARITIES for m in ["precision", "recall", "f1", "accuracy"]]
    metrics_total = {f"{m}": [] for m in metrics_prefixes}

    for cross_idx in range(constants.N_FOLDS)[0:constants.TEST_FOLDS]:
        # Load Data
        train_data = preprocess_data_ACSA(train_dataset[cross_idx], tokenizer)
        test_data = preprocess_data_ACSA(test_dataset[cross_idx], tokenizer)

        # Train Model
        trainer = get_trainer_ACSA(train_data, test_data, tokenizer)
        trainer.train()
