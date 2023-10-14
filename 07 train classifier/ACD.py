from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score
from torch.utils.data import Dataset as TorchDataset
from helper import format_seconds_to_time_string
from transformers import DataCollatorWithPadding
from transformers import set_seed
from scipy.special import expit
import pandas as pd
import numpy as np
import datetime
import warnings
import torch
import json
import time


ASPECT_CATEGORIES = ["GENERAL-IMPRESSION", "FOOD", "SERVICE", "AMBIENCE", "PRICE"]
POLARITIES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
RANDOM_SEED = 43
MODEL_NAME = "deepset/gbert-large"
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 5e-06
MAX_TOKENS = 256

torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
set_seed(RANDOM_SEED)


def aspect_category_labels_to_one_hot(labels):
    one_hot = []
    for label in ASPECT_CATEGORIES:
        if label in labels:
            one_hot.append(1)
        else:
            one_hot.append(0)
    return one_hot


class CustomDatasetACD(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach()
                for key, val in self.encodings.items()}
        item["label"] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)


def preprocess_data_ACD(dataset, tokenizer):
    texts = [example["text"] for example in dataset]
    labels = [list(set([tag["label"] for tag in example["tags"]]))
              for example in dataset]
    labels = [aspect_category_labels_to_one_hot(label) for label in labels]
    labels = torch.tensor(labels, dtype=torch.float32)
    encodings = tokenizer(texts, padding=True, truncation=True,
                          max_length=MAX_TOKENS, return_tensors="pt")
    print(len(encodings), len(encodings[0]), encodings[0])
    return CustomDatasetACD(encodings, labels)


def create_model_ACD():
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        num_labels=len(ASPECT_CATEGORIES),
        problem_type="multi_label_classification"
    ).to(torch.device("cuda" if torch.cuda.is_available() else "mps")) # mps is for apple silicon only!


def compute_metrics_ACD(eval_pred):
    predictions, lab = eval_pred
    predictions = (expit(predictions) > 0.5)
    labels = [l == 1 for l in lab]
    accuracy = accuracy_score(labels, predictions)

    f1_macro = f1_score(labels, predictions, average="macro")
    f1_micro = f1_score(labels, predictions, average="micro")
    f1_weighted = f1_score(labels, predictions, average="weighted")

    class_f1_scores = f1_score(labels, predictions, average=None)

    hamming = hamming_loss(labels, predictions)

    metrics = {
        "hamming_loss": hamming,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "class_f1_scores": class_f1_scores.tolist(),
    }

    return metrics


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

def get_trainer_ACD(model, train_data, test_data, tokenizer):
    # Define Arguments
    training_args = TrainingArgumentsWithMPSSupport(
        output_dir="output",
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
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
        model_init=create_model_ACD,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_ACD
    )

    return trainer
