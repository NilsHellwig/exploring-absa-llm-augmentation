from more_itertools import unique_everseen
import json
from torch.utils.data import Dataset as TorchDataset
import constants


def dict_to_tuple(d):
    return tuple(sorted(d.items()))


def dataset_to_aspect_level_E2E(dataset):
    aspect_dataset = []
    for example in dataset:
        explicit_aspects = [{"start": tag["start"], "end": tag["end"], "polarity": tag["polarity"]} for tag in example["tags"]
                            if tag["type"] == "label-explicit"]

        explicit_aspects = [dict(t) for t in {frozenset(
            d.items()) for d in explicit_aspects}]

        if len(explicit_aspects) > 0:
            aspect_dataset.append({
                "text": example["text"],
                "tags": explicit_aspects,
                "id": example["id"]
            })
    return aspect_dataset


def get_token_role_in_span_E2E(token_start: int, token_end: int, tag: dict):
    tag_start = tag["start"]
    tag_end = tag["end"]
    tag_polarity = tag["polarity"]

    if token_end <= token_start:
        return f"N_{tag_polarity}"
    if token_start < tag_start or token_end > tag_end:
        return f"O_{tag_polarity}"
    if token_start > tag_start:
        return f"I_{tag_polarity}"
    else:
        return f"B_{tag_polarity}"


def preprocess_example_E2E(example, tokenizer):
    input_text = example["text"]
    one_hot_output = [[0 for _ in range(len(constants.LABEL_TO_ID_E2E.keys()) + 1)]
                      for _ in range(constants.MAX_TOKENS_E2E)]
    tokenized_input_text = tokenizer(input_text,
                                     return_offsets_mapping=True,
                                     padding="max_length",
                                     max_length=constants.MAX_TOKENS_E2E,
                                     truncation=True)

    for (token_start, token_end), token_labels in zip(tokenized_input_text["offset_mapping"], one_hot_output):
        found_tags_for_token = 0
        for tag in example["tags"]:
            role = get_token_role_in_span_E2E(token_start, token_end, tag)

            for t in constants.LABEL_TO_ID_E2E:
                if role == t:
                    token_labels[constants.LABEL_TO_ID_E2E[t]] = 1
                    found_tags_for_token+=1
        if found_tags_for_token == 0:
            token_labels[len(token_labels)-1] = 1

    return {
        "input_ids": tokenized_input_text["input_ids"],
        "attention_mask": tokenized_input_text["attention_mask"],
        "offset_mapping": tokenized_input_text["offset_mapping"],
        "labels": one_hot_output
    }


class CustomDatasetE2E(TorchDataset):
    def __init__(self, input_ids, attention_mask, offset_mapping, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.offset_mapping = offset_mapping
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.input_ids[idx]
        item["attention_mask"] = self.attention_mask[idx]
        item["offset_mapping"] = self.offset_mapping[idx]
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def transform_and_save_to_jsonl(dataset, filename):
    transformed_data = []

    for item in dataset:
        transformed_item = {
            "tags": [],
            "id": item["id"],
            "text": item["text"]
        }

        for tag_info in item["tags"]:
            transformed_tag = {
                "end": tag_info["end"],
                "start": tag_info["start"],
                "tag": tag_info["polarity"]
            }
            transformed_item["tags"].append(transformed_tag)

        transformed_data.append(transformed_item)

    with open(filename, 'w', encoding='utf-8') as jsonl_file:
        for item in transformed_data:
            json_line = json.dumps(item, ensure_ascii=False)
            jsonl_file.write(json_line + '\n')


def get_preprocessed_data_E2E(train_data, test_data, tokenizer):
    train_data = dataset_to_aspect_level_E2E(train_data)
    test_data = dataset_to_aspect_level_E2E(test_data)

    train_data = [preprocess_example_E2E(
        example, tokenizer) for example in train_data]
    test_data = [preprocess_example_E2E(
        example, tokenizer) for example in test_data]

    train_data = CustomDatasetE2E([example["input_ids"] for example in train_data],
                                  [example["attention_mask"]
                                      for example in train_data],
                                  [example["offset_mapping"]
                                      for example in train_data],
                                  [example["labels"] for example in train_data])

    test_data = CustomDatasetE2E([example["input_ids"] for example in test_data],
                                 [example["attention_mask"]
                                     for example in test_data],
                                 [example["offset_mapping"]
                                     for example in test_data],
                                 [example["labels"] for example in test_data])


    return train_data, test_data
