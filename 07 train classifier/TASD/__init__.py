from helper import format_seconds_to_time_string
from transformers import AutoTokenizer
from TASD.preprocessing import CustomDatasetTASD, encode_example
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

    tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME_TASD)

    start_time = time.time()

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
        print(train_data[0])

    runtime = time.time() - start_time

    results["runtime"] = runtime
    results["runtime_formatted"] = format_seconds_to_time_string(runtime)

    return results
