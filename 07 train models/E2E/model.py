from transformers import BertPreTrainedModel, BertModel, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from transformers.models.bert.modeling_bert import (BERT_INPUTS_DOCSTRING)
from transformers.utils import (add_start_docstrings_to_model_forward)
from transformers.modeling_outputs import TokenClassifierOutput
from E2E.evaluation import compute_metrics_E2E
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.models.bert.modeling_bert import (
    BERT_INPUTS_DOCSTRING,
    BERT_START_DOCSTRING,
    _CONFIG_FOR_DOC
)


from typing import Optional, Union, Tuple
from transformers import AutoTokenizer
from torch import nn
import numpy as np
import constants
import torch
# source: https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_bert.html
@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForSpanCategorizationE2E(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.init_weights()

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


def create_model_E2E():
    label2id_with_O = constants.LABEL_TO_ID_E2E.copy()
    label2id_with_O["O"] = len(label2id_with_O)

    id2label_with_O = constants.ID_TO_LABEL_E2E.copy()
    id2label_with_O[len(id2label_with_O)] = "O"

    return BertForSpanCategorizationE2E.from_pretrained(constants.MODEL_NAME_E2E, id2label=id2label_with_O, label2id=label2id_with_O)


def get_trainer_E2E(train_data, validation_data, tokenizer, results, cross_idx):
    training_args = TrainingArguments(
        output_dir=constants.OUTPUT_DIR_E2E,
        learning_rate=constants.LEARNING_RATE_E2E,
        per_device_train_batch_size=constants.BATCH_SIZE_E2E,
        per_device_eval_batch_size=constants.BATCH_SIZE_E2E,
        num_train_epochs=constants.EPOCHS_E2E,
        weight_decay=constants.WEIGHT_DECAY_E2E,
        save_strategy="epoch" if constants.EVALUATE_AFTER_EPOCH == True else "no",
        logging_dir="logs",
        logging_steps=100,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        fp16=torch.cuda.is_available(),
        report_to="none",
        do_eval=True if constants.EVALUATE_AFTER_EPOCH == True else False,
        evaluation_strategy="epoch" if constants.EVALUATE_AFTER_EPOCH == True else "no",
        seed=constants.RANDOM_SEED,
    )

    compute_metrics_E2E_fcn = compute_metrics_E2E(results, cross_idx)

    trainer = Trainer(
        model_init=create_model_E2E,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_E2E_fcn,
        callbacks = [EarlyStoppingCallback(early_stopping_patience = constants.N_EPOCHS_EARLY_STOPPING_PATIENCE)]
    )

    return trainer
