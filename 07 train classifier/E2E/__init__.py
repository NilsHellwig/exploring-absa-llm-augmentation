from E2E.preprocessing import get_preprocessed_data_E2E
from transformers import AutoTokenizer
import constants
import time


def train_E2E_model(LLM_NAME, N_REAL, N_SYNTH, TARGET, LLM_SAMPLING, train_dataset, test_dataset):
    results = {
        "LLM_NAME": LLM_NAME,
        "N_REAL": N_REAL,
        "N_SYNTH": N_SYNTH,
        "TARGET": TARGET,
        "LLM_SAMPLING": LLM_SAMPLING,
    }

    metrics_classwise_prefix = ["accuracy", "precision", "recall", "f1"]
    metrics_total_prefix = ["f1_micro", "f1_macro",
                            "precision", "recall", "accuracy"]
    metrics = {f"{metric}": [] for metric in metrics_total_prefix}
    metrics.update({f"{metric}_{ac}": [] for metric in metrics_classwise_prefix for ac in constants.POLARITIES})

    eval_loss = []
    n_samples_train = []
    n_samples_test = []

    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME_E2E)

    start_time = time.time()

    for cross_idx in range(constants.N_FOLDS)[0:constants.TEST_FOLDS]:
        # Load Data
        train_data = train_dataset[cross_idx]
        test_data = test_dataset[cross_idx]

        train_data, test_data = get_preprocessed_data_E2E(train_data, test_data, tokenizer)
