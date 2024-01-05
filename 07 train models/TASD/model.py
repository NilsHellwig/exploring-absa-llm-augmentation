from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from TASD.evaluation import compute_metrics_TASD
from datasets import load_metric
import constants
import torch


def create_model_TASD():
    return AutoModelForSeq2SeqLM.from_pretrained(constants.MODEL_NAME_TASD).to(constants.DEVICE)


def get_trainer_TASD(train_data, valid_data, tokenizer, results, cross_idx):
    args = Seq2SeqTrainingArguments(
        output_dir=constants.OUTPUT_DIR_TASD+"_" +
        results["LLM_NAME"]+"_"+str(results["N_REAL"])+"_"+str(results["N_SYNTH"]) +
        "_"+results["TARGET"]+"_"+results["LLM_SAMPLING"],
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

    compute_metrics_TASD_fcn = compute_metrics_TASD(results, cross_idx)

    trainer = Seq2SeqTrainer(
        model_init=create_model_TASD,
        args=args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_TASD_fcn,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=constants.N_EPOCHS_EARLY_STOPPING_PATIENCE)]
    )
    return trainer
