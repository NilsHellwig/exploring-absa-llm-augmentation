from torch.utils.data import Dataset as TorchDataset
import constants


def dataset_to_aspect_level_OTE(dataset):
    aspect_dataset = []
    for example in dataset:
        aspect_categories = list(
            set([tag["label"] for tag in example["tags"] if tag["type"] == "label-explicit"]))
        for ac in aspect_categories:
            aspect_dataset.append({
                "text": example["text"],
                "aspect_category": ac,
                "tags": [tag for tag in example["tags"] if tag["type"] == "label-explicit" and tag["label"] == ac],
                "id": example["id"]
            })
    return aspect_dataset


def get_token_role_in_span_OTE(token_start: int, token_end: int, span_start: int, span_end: int):
    if token_end <= token_start:
        return "N"
    if token_start < span_start or token_end > span_end:
        return "O"
    if token_start > span_start:
        return "I"
    else:
        return "B"


def preprocess_example_OTE(example, tokenizer):
    input_text = example["text"] + "[SEP]" + example["aspect_category"]
    one_hot_output = [[0 for _ in constants.LABEL_TO_ID_OTE.keys()]
                      for _ in range(constants.MAX_TOKENS_OTE)]

    tokenized_input_text = tokenizer(input_text,
                                     return_offsets_mapping=True,
                                     padding="max_length",
                                     max_length=constants.MAX_TOKENS_OTE,
                                     truncation=True)

    for (token_start, token_end), token_labels in zip(tokenized_input_text["offset_mapping"], one_hot_output):
        for span in example["tags"]:
            role = get_token_role_in_span_OTE(
                token_start, token_end, span["start"], span["end"])
            if role == "B":
                token_labels[constants.LABEL_TO_ID_OTE["B"]] = 1
            elif role == "I":
                token_labels[constants.LABEL_TO_ID_OTE["I"]] = 1

        if token_labels[constants.LABEL_TO_ID_OTE["B"]] == 0 and token_labels[constants.LABEL_TO_ID_OTE["I"]] == 0:
            token_labels[constants.LABEL_TO_ID_OTE["O"]] = 1


    return {
        "input_ids": tokenized_input_text["input_ids"],
        "attention_mask": tokenized_input_text["attention_mask"],
        "offset_mapping": tokenized_input_text["offset_mapping"],
        "aspect_category": example["aspect_category"],
        "labels": one_hot_output
    }


class CustomDatasetOTE(TorchDataset):
    def __init__(self, input_ids, attention_mask, offset_mapping, aspect_categories, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.offset_mapping = offset_mapping
        self.aspect_categories = aspect_categories
        self.labels = labels

    def __getitem__(self, idx):
        # item = {key: val[idx].clone().detach() for key, val in self.input_ids.items()}
        item = {}
        item["input_ids"] = self.input_ids[idx]
        item["attention_mask"] = self.attention_mask[idx]
        item["offset_mapping"] = self.offset_mapping[idx]
        item["aspect_category"] = self.aspect_categories[idx]
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def get_preprocessed_data_OTE(train_data, test_data, tokenizer):
    train_data = dataset_to_aspect_level_OTE(train_data)
    test_data = dataset_to_aspect_level_OTE(test_data)
    train_data = [preprocess_example_OTE(
        example, tokenizer) for example in train_data]
    test_data = [preprocess_example_OTE(
        example, tokenizer) for example in test_data]
    train_data = CustomDatasetOTE([example["input_ids"] for example in train_data],
                                  [example["attention_mask"]
                                      for example in train_data],
                                  [example["offset_mapping"]
                                      for example in train_data],
                                  [example["aspect_category"]
                                      for example in train_data],
                                  [example["labels"] for example in train_data])
    test_data = CustomDatasetOTE([example["input_ids"] for example in test_data],
                                 [example["attention_mask"]
                                     for example in test_data],
                                 [example["offset_mapping"]
                                     for example in test_data],
                                 [example["aspect_category"]
                                     for example in test_data],
                                 [example["labels"] for example in test_data])
    return train_data, test_data
