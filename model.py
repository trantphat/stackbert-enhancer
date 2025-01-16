import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import List, Optional, Tuple, Union


class BertCustomBinaryClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCustomBinaryClassifier, self).__init__(config)

        # Initialize BERT model
        self.bert = BertModel(config)

        # Define layers
        self.dropout = nn.Dropout(0.30)
        self.hidden_layer = nn.Linear(768, 32)
        self.tanh_activation = nn.Tanh()
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid_activation = nn.Sigmoid()

    def forward(
        self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, kmer=3, return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # Decide whether to return a dictionary or tuple
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input through BERT model
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)

        # Extract hidden states from the last layer
        # Shape: (batch size, N + 2, 768) including [CLS] and [SEP] tokens or (batch_size, sequence_length, hidden_size)
        last_hidden_states = bert_outputs.last_hidden_state
        # pooler_output = bert_outputs.pooler_output  # Shape: (1, 768) [CLS] token

        # Exclude [CLS] and [SEP] tokens and calculate the average of token embeddings
        token_embeddings = last_hidden_states[:, 1 : 202 - kmer]
        averaged_embeddings = token_embeddings.mean(dim=-2)  # Shape of averaged_token_embeddings is (1, 768) or (batch_size, hidden_size)

        # Apply dropout to the averaged embeddings
        dropout_output = self.dropout(averaged_embeddings)

        # Pass through the hidden layer and apply Tanh function
        hidden_output = self.hidden_layer(dropout_output)
        tanh_output = self.tanh_activation(hidden_output)

        # Pass through the output layer and apply Sigmoid function
        final_output = self.output_layer(tanh_output)
        logits = self.sigmoid_activation(final_output)

        loss = None

        # https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_bert.html
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py

        # Compute binary cross-entropy loss if labels are provided
        if labels is not None:
            flattened_labels = labels.view(-1).to(torch.float32)
            loss_function = nn.BCEWithLogitsLoss()
            loss = loss_function(logits.view(-1), flattened_labels)

        # Return outputs as a tuple or dictionary based on `return_dict`
        if not return_dict:
            output = (logits,) + bert_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.last_hidden_state,
            attentions=None,
        )
