from transformers import TrainingArguments, Trainer
from transformers.models.bert.modeling_bert import (BERT_INPUTS_DOCSTRING)
from transformers.utils import (add_start_docstrings_to_model_forward)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import BertModel, BertPreTrainedModel
from OTE.evaluation import compute_metrics_OTE
from transformers import DataCollatorWithPadding
from typing import Optional, Union, Tuple
from torch.nn.functional import softmax
from torch import nn
import constants
import torch


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

        # Here, I added softmax...
        logits = softmax(logits, dim=2)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
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
    return BertForSpanCategorizationOTE.from_pretrained(constants.MODEL_NAME_OTE, id2label=constants.ID_TO_LABEL_OTE, label2id=constants.LABEL_TO_ID_OTE)


def get_trainer_OTE(train_data, test_data, tokenizer):
    training_args = TrainingArguments(
        output_dir=constants.OUTPUT_DIR_OTE,
        learning_rate=constants.LEARNING_RATE_OTE,
        per_device_train_batch_size=constants.BATCH_SIZE_OTE,
        per_device_eval_batch_size=constants.BATCH_SIZE_OTE,
        num_train_epochs=constants.EPOCHS_OTE,
        weight_decay=constants.WEIGHT_DECAY_OTE,
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

    compute_metrics_OTE_fcn = compute_metrics_OTE(test_data)

    trainer = Trainer(
        model_init=create_model_OTE,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_OTE_fcn
    )

    return trainer
