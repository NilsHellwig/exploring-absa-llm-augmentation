from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from TASD.evaluation import compute_metrics
from datasets import load_metric
import constants
import torch


def create_model_TASD():
    return AutoModelForSeq2SeqLM.from_pretrained(constants.MODEL_NAME_TASD).to("cuda")


def get_trainer_TASD(train_data, test_data, tokenizer):
    args = Seq2SeqTrainingArguments(
        output_dir=constants.OUTPUT_DIR_TASD,
        logging_strategy=constants.LOGGING_STRATEGY_TASD,
        save_strategy="epoch" if constants.EVALUATE_AFTER_EPOCH == True else "no",
        learning_rate=constants.LEARNING_RATE_TASD,
        num_train_epochs=constants.EPOCHS_TASD,
        per_device_train_batch_size=constants.BATCH_SIZE_TASD,
        per_device_eval_batch_size=constants.BATCH_SIZE_TASD,
        predict_with_generate=True,
        load_best_model_at_end=True,
        weight_decay=constants.WEIGHT_DECAY_TASD,
        metric_for_best_model=constants.METRIC_FOR_BEST_MODEL_TASD,
        seed=constants.RANDOM_SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
        do_eval=True if constants.EVALUATE_AFTER_EPOCH == True else False,
        evaluation_strategy="epoch" if constants.EVALUATE_AFTER_EPOCH == True else "no",
        generation_max_length=256
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    trainer = Seq2SeqTrainer(
        model_init=create_model_TASD,
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    return trainer
