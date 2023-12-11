import constants
import random
import json

# This function double-checks if there are any training examples that are also used for testing
# - this should for sure not be the case, however I'm checking it here once more


def check_for_duplicate_ids(train_dataset, test_dataset, validation_dataset):
    for split_id in range(len(train_dataset)):
        ids_train = [example["id"] for example in train_dataset[split_id]]
        ids_test = [example["id"] for example in test_dataset[split_id]]
        ids_validation = [example["id"]
                          for example in validation_dataset[split_id]]

        # Check for duplicates within each dataset
        if len(ids_train) != len(set(ids_train)):
            raise ValueError("Duplicates found in training data.")

        if len(ids_test) != len(set(ids_test)):
            raise ValueError("Duplicates found in test data.")

        if len(ids_validation) != len(set(ids_validation)):
            raise ValueError("Duplicates found in validation data.")

        # Check for common ids across datasets
        common_ids = set(ids_train) & set(ids_test) & set(ids_validation)

        if len(common_ids):
            raise ValueError(
                "Ids used in both training, test, or validation data: " + str(common_ids))


def load_dataset_folds(LLM_NAME, N_REAL, N_SYNTH, LLM_SAMPLING):
    if LLM_SAMPLING == "fixed" and N_SYNTH == 0:
        raise Exception(
            "Fixed adds 25 real samples. Please use random for LLM_SAMPLING to evaluate only real examples.")
    if LLM_SAMPLING == "fixed" and N_SYNTH not in [475, 975, 1975]:
        raise Exception(
            "If Few-Shot Examples are fixed, N_SYNTH needs to be 475, 975, 1975")

    if LLM_SAMPLING == "random" and N_SYNTH not in [0, 500, 1000, 1500]:
        raise Exception(
            "If Few-Shot Examples are fixed, N_SYNTH needs to be 0, 500, 1000 or 1500")

    # 2. Load Test Data
    test_dataset = []
    for i in range(1, constants.N_FOLDS+1):
        with open(f'../07 train models/real/split_{constants.SPLIT_LOOP[i]}.json', 'r') as json_datei:
            real_split = json.load(json_datei)
            random.shuffle(real_split)
            test_dataset.append(real_split)

    # 2. Load Test Data
    validation_dataset = []
    for i in range(2, constants.N_FOLDS+2):
        with open(f'../07 train models/real/split_{constants.SPLIT_LOOP[i]}.json', 'r') as json_datei:
            real_split = json.load(json_datei)
            random.shuffle(real_split)
            validation_dataset.append(real_split)

    # 3. Load Real Train Data
    train_dataset = []

    # 3.1 Add 25 real samples if condition is fixed examples
    if LLM_SAMPLING == "fixed" and N_REAL == 25:
        for split_idx in constants.SPLIT_LOOP[0:constants.N_FOLDS]:
            with open(f'../07 train models/real_fixed/split_{split_idx}.json', 'r') as json_datei:
                real_split = json.load(json_datei)
                random.shuffle(real_split)
                train_dataset.append(real_split)
    # This is for the language analysis - (we also need the opportunity to only load the synthetic fixed examples)
    if LLM_SAMPLING == "fixed" and N_REAL == 0:
        for split_idx in constants.SPLIT_LOOP[0:constants.N_FOLDS]:
            train_dataset.append([])

    # 3.2 Add real samples if condtion is random or N_SYNTH = 0 (condtions without llms)
    if LLM_SAMPLING == "random" or N_SYNTH == 0:
        # Wenn N_REAL = 1000, dann ist n_splits_required_real = 2
        n_splits_required_real = constants.N_SPLITS_MAP_REAL.get(N_REAL, 0)

        # load first split
        for idx_real in range(constants.N_FOLDS):
            train_dataset.append([])
            with open(f'../07 train models/real/split_{idx_real}.json', 'r') as json_datei:
                real_split = json.load(json_datei)
            for example in real_split:
                train_dataset[idx_real].append(example)
            random.shuffle(train_dataset[idx_real])

        # load more splits if required
        if n_splits_required_real > 1:
            for idx_real in range(constants.N_FOLDS):
                real_data_splits = constants.SPLIT_LOOP[2 +
                                                        idx_real: 2 + idx_real + n_splits_required_real - 1]
                for split in real_data_splits:
                    with open(f'../07 train models/real/split_{split}.json', 'r') as json_datei:
                        real_split = json.load(json_datei)
                    for example in real_split:
                        train_dataset[idx_real].append(example)
                random.shuffle(train_dataset[idx_real])

    # 4. Load Synth Train Data
    if N_SYNTH > 0:
        for split_id in range(constants.N_FOLDS):
            with open(f'../07 train models/synth/{LLM_NAME}/{LLM_SAMPLING}/split_{constants.SPLIT_LOOP[split_id]}.json', 'r') as json_datei:
                synth_split = json.load(json_datei)[:N_SYNTH]
                for example in synth_split:
                    train_dataset[split_id].append(example)
            random.shuffle(train_dataset[split_id])

    print("Train:", len(train_dataset[0]), len(train_dataset))
    print("Test:", len(test_dataset[0]), len(test_dataset))
    print("Validation:", len(validation_dataset[0]), len(validation_dataset))

    check_for_duplicate_ids(train_dataset, test_dataset, validation_dataset)

    return train_dataset, test_dataset, validation_dataset
