"""
Implementation borrowed from transformers package and extended to support multiple prediction heads:

https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
"""

from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import transformers
from transformers.models.bert.tokenization_bert_fast import BertTokenizer
# from transformers import models
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.megatron_bert.modeling_megatron_bert import (
    MegatronBertPreTrainedModel,
    MegatronBertModel,
    MEGATRON_BERT_INPUTS_DOCSTRING
)
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BERT_INPUTS_DOCSTRING,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    BertModel,
    # SequenceClassification docstring
    _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
    _SEQ_CLASS_EXPECTED_OUTPUT,
    _SEQ_CLASS_EXPECTED_LOSS,
)

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)

class MegatronBertForSequenceClassificationModified(MegatronBertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = kwargs.get("task_labels_map", {})
        self.config = config
        
        self.bert = MegatronBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(
            config.hidden_size, list(self.num_labels.values())[0]
        )
        self.classifier2 = nn.Linear(
            config.hidden_size, list(self.num_labels.values())[1]
        )
        self.classifier3 = nn.Linear(
            config.hidden_size, list(self.num_labels.values())[2]
        )

        # Initialize weights and apply final processing
        self.post_init()

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
        task_name=None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        labels = kwargs.pop("label_ids")
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
        # pooled_output = outputs[1]
        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)
        logits = None
        if task_name == list(self.num_labels.keys())[0]:
            logits = self.classifier1(pooled_output)
        elif task_name == list(self.num_labels.keys())[1]:
            logits = self.classifier2(pooled_output)
        elif task_name == list(self.num_labels.keys())[2]:
            logits = self.classifier3(pooled_output)
        else:
            raise Exception("task " + task_name + " not known.")

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels[task_name] == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels[task_name] > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels[task_name] == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels[task_name]), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = kwargs.get("task_labels_map", {})
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        ## add task specific output heads
        self.classifier1 = nn.Linear(
            config.hidden_size, list(self.num_labels.values())[0]
        )
        self.classifier2 = nn.Linear(
            config.hidden_size, list(self.num_labels.values())[1]
        )
        self.classifier3 = nn.Linear(
            config.hidden_size, list(self.num_labels.values())[2]
        )

        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task_name=None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = None
        if task_name == list(self.num_labels.keys())[0]:
            logits = self.classifier1(pooled_output)
        elif task_name == list(self.num_labels.keys())[1]:
            logits = self.classifier2(pooled_output)
        elif task_name == list(self.num_labels.keys())[2]:
            logits = self.classifier3(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels[task_name] == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels[task_name] > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels[task_name] == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels[task_name]), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
