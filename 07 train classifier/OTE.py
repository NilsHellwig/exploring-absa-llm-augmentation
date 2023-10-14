from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers.models.bert.modeling_bert import (BERT_INPUTS_DOCSTRING)
from transformers.utils import (add_start_docstrings_to_model_forward)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import BertModel, BertPreTrainedModel
from sklearn.metrics import multilabel_confusion_matrix
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorWithPadding
from helper import format_seconds_to_time_string
from typing import Optional, Union, Tuple
from torch import nn
import numpy as np
import torch

label2id = {
    'O': 0,
    'B': 1,
    'I': 2,
}
id2label = {
    0: 'O',
    1: 'B',
    2: 'I',
}

n_labels = len(id2label)

MODEL_NAME = "deepset/gbert-base"
MAX_TOKENS = 256
RANDOM_SEED = 43
BATCH_SIZE = 16
N_EPOCHS = 1
LEARNING_RATE = 5e-06


def dataset_to_aspect_level_OTE(dataset):
    aspect_dataset = []
    for example in dataset:
        aspect_categories = list(
            set([tag["label"] for tag in example["tags"] if tag["type"] == "label-explicit"]))
        for ac in aspect_categories:
            aspect_dataset.append({
                "text": example["text"],
                "aspect_category": ac,
                "tags": [tag for tag in example["tags"] if tag["type"] == "label-explicit"],
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
    one_hot_output = [[0 for _ in label2id.keys()] for _ in range(MAX_TOKENS)]

    tokenized_input_text = tokenizer(input_text,
                                     return_offsets_mapping=True,
                                     padding="max_length",
                                     max_length=MAX_TOKENS,
                                     truncation=True)

    for (token_start, token_end), token_labels in zip(tokenized_input_text["offset_mapping"], one_hot_output):
        for span in example["tags"]:
            role = get_token_role_in_span_OTE(
                token_start, token_end, span["start"], span["end"])
            if role == "B":
                token_labels[label2id["B"]] = 1
            elif role == "I":
                token_labels[label2id["I"]] = 1

    return {
        "input_ids": tokenized_input_text["input_ids"],
        "attention_mask": tokenized_input_text["attention_mask"],
        "offset_mapping": tokenized_input_text["offset_mapping"],
        "labels": one_hot_output
    }


class CustomDatasetOTE(TorchDataset):
    def __init__(self, input_ids, attention_mask, offset_mapping, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.offset_mapping = offset_mapping
        self.labels = labels

    def __getitem__(self, idx):
        # item = {key: val[idx].clone().detach() for key, val in self.input_ids.items()}
        item = {}
        item["input_ids"] = self.input_ids[idx]
        item["attention_mask"] = self.attention_mask[idx]
        item["offset_mapping"] = self.offset_mapping[idx]
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
                                  [example["labels"] for example in train_data])
    test_data = CustomDatasetOTE([example["input_ids"] for example in test_data],
                                 [example["attention_mask"]
                                     for example in test_data],
                                 [example["offset_mapping"]
                                     for example in test_data],
                                 [example["labels"] for example in test_data])
    return train_data, test_data


def divide(a: int, b: int):
    return a / b if b > 0 else 0


def compute_metrics_OTE(p):
    predictions, true_labels = p
    predicted_labels = np.where(predictions > 0, np.ones(
        predictions.shape), np.zeros(predictions.shape))
    metrics = {}
    cm = multilabel_confusion_matrix(
        true_labels.reshape(-1, n_labels), predicted_labels.reshape(-1, n_labels))

    for label_idx, matrix in enumerate(cm):
        if label_idx == 0:
            continue  # We don't care about the label "O"
        tp, fp, fn = matrix[1, 1], matrix[0, 1], matrix[1, 0]
        precision = divide(tp, tp + fp)
        recall = divide(tp, tp + fn)
        f1 = divide(2 * precision * recall, precision + recall)
        metrics[f"f1_{id2label[label_idx]}"] = f1

    f1_micro = sum(list(metrics.values())) / (n_labels - 1)
    metrics["f1_micro"] = f1_micro

    return metrics


class BertForSpanCategorizationOTE(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_model_OTE():
    return BertForSpanCategorizationOTE.from_pretrained(MODEL_NAME, id2label=id2label, label2id=label2id)


# Source: https://github.com/huggingface/transformers/issues/17971
class TrainingArgumentsWithMPSSupport(TrainingArguments):

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

def get_trainer_OTE(train_data, test_data, tokenizer):
    training_args = TrainingArgumentsWithMPSSupport(
        output_dir="output2",
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=N_EPOCHS,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="logs",
        logging_steps=100,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model_init=create_model_OTE,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_OTE
    )

    return trainer
