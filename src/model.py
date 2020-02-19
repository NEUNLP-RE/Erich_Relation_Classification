from __future__ import absolute_import, division, print_function

from pytorch_transformers import BertPreTrainedModel, BertModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


class BertForRelExtractionClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """

    def __init__(self, config):
        super(BertForRelExtractionClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.entity_activate_fn = nn.Tanh()
        self.entity_ffn = nn.Linear(config.hidden_size, config.hidden_size)
        self.final_hidden_ffn = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(3 * config.hidden_size, self.config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids, entity_a, entity_b, token_type_ids, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]
        bsz, length, hidden = sequence_output.size()
        avg_entity_a = sequence_output.new_full([bsz, hidden], 0.0)
        avg_entity_b = sequence_output.new_full([bsz, hidden], 0.0)
        for i in range(bsz):
            avg_entity_a[i] = torch.sum(sequence_output[i, entity_a[i][0]:entity_a[i][1] + 1, :], 0) \
                       / float(entity_a[i][1] - entity_a[i][0] + 1)
            avg_entity_b[i] = torch.sum(sequence_output[i, entity_b[i][0]:entity_b[i][1] + 1, :], 0) \
                       / float(entity_b[i][1] - entity_b[i][0] + 1)

        # entity state ffn
        h1 = self.dropout(self.entity_activate_fn(avg_entity_a))
        h1 = self.entity_ffn(h1)
        h2 = self.dropout(self.entity_activate_fn(avg_entity_b))
        h2 = self.entity_ffn(h2)
        # hidden state ffn
        h0 = sequence_output[:, 0, :]
        h0 = self.dropout(self.entity_activate_fn(h0))
        h0 = self.final_hidden_ffn(h0)

        h_concat = self.dropout(torch.cat((h1, h2, h0), -1))
        logits = self.classifier(h_concat)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
