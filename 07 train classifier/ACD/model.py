from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from ACD.evaluation import compute_metrics_ACD
import constants
import torch


def create_model_ACD():
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=constants.MODEL_NAME_ACD,
        num_labels=len(constants.ASPECT_CATEGORIES),
        problem_type="multi_label_classification"
    ).to(torch.device(constants.DEVICE))



def get_trainer_ACD(train_data, test_data, tokenizer):
    # Define Arguments
    training_args = TrainingArguments(
        output_dir=constants.OUTPUT_DIR_ACD,
        learning_rate=constants.LEARNING_RATE_ACD,
        num_train_epochs=constants.EPOCHS_ACD,
        per_device_train_batch_size=constants.BATCH_SIZE_ACD,
        per_device_eval_batch_size=constants.BATCH_SIZE_ACD,
        evaluation_strategy=constants.EVALUATION_STRATEGY_ACD,
        save_strategy="epoch",
        logging_dir="logs",
        logging_steps=100,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=constants.RANDOM_SEED,
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
