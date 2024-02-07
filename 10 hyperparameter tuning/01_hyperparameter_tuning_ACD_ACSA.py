import random
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from scipy.special import expit
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorWithPadding
import optuna
import concurrent.futures
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score
import warnings
import sys
import shutil
import subprocess
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="transformers.optimization")


class CustomDataset(TorchDataset):
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


class MultiLabelABSA:
    def __init__(self, data_path, result_path, model_name, n_samples):
        self.data_path = data_path
        self.result_path = result_path
        self.model_id = model_name
        self.n_samples = n_samples

    def preprocess_data(self, data, tokenizer):
        texts = data["text"].tolist()
        labels = data.iloc[:, 1:].astype(float).values.tolist()
        labels = torch.tensor(labels, dtype=torch.float32)
        encodings = tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        return CustomDataset(encodings, labels)

    def create_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            num_labels=self.data.shape[1] - 1,
            problem_type="multi_label_classification"
        )
        return model

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred

        predictions = (expit(predictions) > 0.5).astype(int)

        accuracy = accuracy_score(labels, predictions)

        f1_macro = f1_score(labels, predictions,
                            average="macro", zero_division=0)
        f1_micro = f1_score(labels, predictions,
                            average="micro", zero_division=0)
        f1_weighted = f1_score(labels, predictions,
                               average="weighted", zero_division=0)

        class_f1_scores = f1_score(
            labels, predictions, average=None, zero_division=0)

        hamming = hamming_loss(labels, predictions)

        return {
            "hamming_loss": hamming,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "class_f1_scores": class_f1_scores.tolist(),
        }

    def split_train_test(self):
        shuffled_indices = self.data.index.tolist()
        random.shuffle(shuffled_indices)

        train_indices = shuffled_indices[:2000]
        train_data = self.data.loc[train_indices][0:self.n_samples]

        test_indices = shuffled_indices[2000:2500]
        test_data = self.data.loc[test_indices]

        print("n samples_train", len(train_data), len(test_data))
        return train_data, test_data

    def objective(self, trial):
        learning_rate = trial.suggest_categorical(
            "learning_rate", self.hyperparameters["learning_rate"])
        num_train_epochs = trial.suggest_int(
            "num_train_epochs", self.hyperparameters["epochs"][0], self.hyperparameters["epochs"][1])
        per_device_train_batch_size = trial.suggest_categorical(
            "per_device_train_batch_size", self.hyperparameters["batch_size"])

        f1_micro_scores = []
        f1_macro_scores = []
        f1_weighted_scores = []
        accuracy_scores = []
        class_f1_scores = []
        loss = []
        hamming = []

        # Start measuring the runtime
        start_time = time.time()

        random.seed(43)

        for k in range(5):
            train_data, test_data = self.split_train_test()
            train_dataset = self.preprocess_data(
                train_data[0:self.n_samples], self.tokenizer)
            test_dataset = self.preprocess_data(test_data, self.tokenizer)

            model = self.create_model()

            training_args = TrainingArguments(
                output_dir=f"output_{sys.argv[2]}",
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_train_batch_size,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_dir="logs",
                logging_steps=100,
                logging_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1_micro",
                fp16=True,
                report_to="none"
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics
            )

            print("Using the following hyperparameters: lr=" + str(learning_rate) + " - epochs=" +
                  str(num_train_epochs) + " - batch=" + str(per_device_train_batch_size))

            trainer.train()
            eval_metrics = trainer.evaluate()

            f1_micro_scores.append(eval_metrics["eval_f1_micro"])
            f1_macro_scores.append(eval_metrics["eval_f1_macro"])
            f1_weighted_scores.append(eval_metrics["eval_f1_weighted"])
            accuracy_scores.append(eval_metrics["eval_accuracy"])
            class_f1_scores.append(eval_metrics["eval_class_f1_scores"])
            loss.append(eval_metrics["eval_loss"])
            hamming.append(eval_metrics["eval_hamming_loss"])

        shutil.rmtree(f"output_{sys.argv[2]}")
        subprocess.call("rm -rf /home/mi/.local/share/Trash", shell=True)

        # Calculate runtime
        runtime = time.time() - start_time

        # Store the results in the DataFrame
        self.results_df.loc[len(self.results_df)] = [
            trial.number,
            learning_rate,
            num_train_epochs,
            per_device_train_batch_size,
            runtime,
            np.mean(loss),
            np.mean(hamming),
            np.mean(accuracy_scores),
            np.mean(f1_micro_scores),
            np.mean(f1_macro_scores),
            np.mean(f1_weighted_scores),
            [sum(col) / len(col) for col in zip(*class_f1_scores)],
        ]

        # Save the results as a TSV file
        self.results_df.to_csv(self.result_path, sep="\t", index=False)

        return np.mean(loss)

    def hyperparameterSearch(self, hp_config):
        # Load data
        self.hyperparameters = hp_config

        self.data = pd.read_csv(
            self.data_path, delimiter="\t", index_col=0).reset_index(drop=True)
        self.data.columns = ["text"] + \
            [f"aspect_{i}" for i in range(1, self.data.shape[1])]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Update the results_df DataFrame
        self.results_df = pd.DataFrame(columns=["trial", "learning_rate", "num_train_epochs", "per_device_train_batch_size", "runtime",
                                                "loss", "hamming_loss", "accuracy", "f1_micro", "f1_macro", "f1_weighted", "class_f1_scores"])

        # Optuna optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(
            self.objective, n_trials=self.hyperparameters['num_trials'])


print(sys.argv[1], sys.argv[2])

n_samples = int(sys.argv[1])

if sys.argv[2] == "ACSA":
    result_path = f"optuna_20_gbert_acsa_{n_samples}.tsv"
    data_path = "./datasets/fehle-2023-hotel-absa/complete_re_df_cat_pol.tsv"

if sys.argv[2] == "ACD":
    result_path = f"optuna_20_gbert_acd_{n_samples}.tsv"
    data_path = "./datasets/fehle-2023-hotel-absa/complete_re_df_cat.tsv"


model_name = "deepset/gbert-large"

hyperparameters = {
    "num_trials": 20,
    "epochs": [2, 20],
    "batch_size": [8, 16, 32],
    "learning_rate": [2e-5, 3e-5, 4e-5, 5e-5]
}

# deepset/gbert-large
# dbmdz/bert-base-german-uncased
# distilbert-base-german-cased

absa = MultiLabelABSA(data_path, result_path, model_name, n_samples)
absa.hyperparameterSearch(hyperparameters)
