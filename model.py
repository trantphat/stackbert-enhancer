import torch
from torch import nn
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
        token_embeddings = last_hidden_states[:, 1 : 200 + 2 - kmer]  # Shape: (batch_size, num_tokens, hidden_size)
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

        # Return outputs as a tuple or dictionary
        if not return_dict:
            output = (logits,) + bert_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.last_hidden_state,
            attentions=None,
        )


class BertAttentionScoreExtractor(BertPreTrainedModel):
    def __init__(self, config):
        super(BertAttentionScoreExtractor, self).__init__(config)

        # Initialize BERT model
        self.bert = BertModel(config)

        # Define layers
        self.dropout = nn.Dropout(0.30)
        self.hidden_layer = nn.Linear(768, 32)
        self.tanh_activation = nn.Tanh()
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid_activation = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        attention_scores = outputs.attentions  # (batch_size, num_heads, sequence_length, sequence_length)
        return attention_scores


class BertCustomBinaryClassifierWithCNNV2(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCustomBinaryClassifierWithCNNV2, self).__init__(config)

        # Initialize BERT model
        self.bert = BertModel(config)

        # Define layers
        self.dropout = nn.Dropout(0.30)

        # Define a single-layer CNN
        self.conv1d = nn.Conv1d(in_channels=768, out_channels=32, kernel_size=3)

        self.relu_activation = nn.ReLU()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid_activation = nn.Sigmoid()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        kmer=3,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        # Decide whether to return a dictionary or tuple
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input through BERT model
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)

        # Extract hidden states from the last layer
        last_hidden_states = bert_outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

        token_embeddings = last_hidden_states[:, :-1, :]  # Shape: (batch_size, num_tokens-1, hidden_size)

        # Transpose for CNN input (batch_size, hidden_size, num_tokens)
        cnn_input = token_embeddings.permute(0, 2, 1)

        # Apply CNN layer
        cnn_output = self.conv1d(cnn_input)  # Shape: (batch_size, out_channels, num_tokens_after_conv)

        # Apply ReLU activation
        relu_output = self.relu_activation(cnn_output)

        # Global max pooling over the sequence length dimension (num_tokens_after_conv)
        pooled_output = torch.max(relu_output, dim=2).values  # Shape: (batch_size, out_channels)

        # Apply dropout to pooled output
        dropout_output = self.dropout(pooled_output)

        # Pass through the first fully connected layer
        fc1_features = self.fc1(dropout_output)

        # Apply dropout to the features from fc1
        fc1_features_dropout = self.dropout(fc1_features)

        # Pass through the second fully connected layer
        fc2_output = self.fc2(fc1_features_dropout)

        # Apply sigmoid activation for binary classification
        logits = self.sigmoid_activation(fc2_output)  # Shape: (batch_size, 1)

        loss = None

        # Compute binary cross-entropy loss if labels are provided
        if labels is not None:
            flattened_labels = labels.view(-1).to(torch.float32)
            loss_function = nn.BCEWithLogitsLoss()
            loss = loss_function(logits.view(-1), flattened_labels)

        # Return outputs as a tuple or dictionary
        if not return_dict:
            output = (logits,) + bert_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.last_hidden_state,
            attentions=None,
        )


class BertCustomBinaryClassifierWithCNN(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCustomBinaryClassifierWithCNN, self).__init__(config)

        # Initialize BERT model
        self.bert = BertModel(config)

        # Define layers
        self.dropout = nn.Dropout(0.30)

        # Define a single-layer CNN
        self.conv1d = nn.Conv1d(in_channels=768, out_channels=64, kernel_size=3)  # Input size matches BERT's hidden size  # Number of filters  # Kernel size for the convolution

        self.relu_activation = nn.ReLU()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid_activation = nn.Sigmoid()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        kmer=3,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        # Decide whether to return a dictionary or tuple
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input through BERT model
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)

        # Extract hidden states from the last layer
        last_hidden_states = bert_outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

        # Exclude [CLS] and [SEP] tokens and adjust for k-mer size
        token_embeddings = last_hidden_states[:, 1 : 200 + 2 - kmer]  # Shape: (batch_size, num_tokens, hidden_size)

        # Transpose for CNN input (batch_size, hidden_size, num_tokens)
        cnn_input = token_embeddings.permute(0, 2, 1)

        # Apply CNN layer
        cnn_output = self.conv1d(cnn_input)  # Shape: (batch_size, out_channels, num_tokens_after_conv)

        # Apply ReLU activation
        relu_output = self.relu_activation(cnn_output)

        # Global max pooling over the sequence length dimension (num_tokens_after_conv)
        pooled_output = torch.max(relu_output, dim=2).values  # Shape: (batch_size, out_channels)

        # Apply dropout to pooled output
        dropout_output = self.dropout(pooled_output)

        # Pass through the first fully connected layer
        fc1_features = self.fc1(dropout_output)

        # Apply dropout to the features from fc1
        fc1_features_dropout = self.dropout(fc1_features)

        # Pass through the second fully connected layer
        fc2_output = self.fc2(fc1_features_dropout)

        # Apply sigmoid activation for binary classification
        logits = self.sigmoid_activation(fc2_output)  # Shape: (batch_size, 1)

        loss = None

        # Compute binary cross-entropy loss if labels are provided
        if labels is not None:
            flattened_labels = labels.view(-1).to(torch.float32)
            loss_function = nn.BCEWithLogitsLoss()
            loss = loss_function(logits.view(-1), flattened_labels)

        # Return outputs as a tuple or dictionary
        if not return_dict:
            output = (logits,) + bert_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.last_hidden_state,
            attentions=None,
        )


class BertLogitsExtractor(BertPreTrainedModel):
    def __init__(self, config):
        super(BertLogitsExtractor, self).__init__(config)

        # Initialize BERT model
        self.bert = BertModel(config)

        # Define layers (only the ones needed to get to the logits)
        self.dropout = nn.Dropout(0.30)
        self.hidden_layer = nn.Linear(768, 32)
        self.tanh_activation = nn.Tanh()
        self.output_layer = nn.Linear(32, 1)

    def forward(
        self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, kmer=3, return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # Decide whether to return a dictionary or tuple
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input through BERT model
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)

        # Extract hidden states from the last layer
        last_hidden_states = bert_outputs.last_hidden_state

        # Exclude [CLS] and [SEP] tokens and calculate the average of token embeddings
        token_embeddings = last_hidden_states[:, 1 : 200 + 2 - kmer]
        averaged_embeddings = token_embeddings.mean(dim=-2)

        # Apply dropout to the averaged embeddings
        dropout_output = self.dropout(averaged_embeddings)

        # Pass through the hidden layer and apply Tanh function
        hidden_output = self.hidden_layer(dropout_output)
        tanh_output = self.tanh_activation(hidden_output)

        # Pass through the output layer (without Sigmoid)
        logits = self.output_layer(tanh_output)

        # Return outputs as a tuple or dictionary
        if not return_dict:
            output = (logits,) + bert_outputs[2:]
            return output

        return SequenceClassifierOutput(
            loss=None,  # No loss calculation
            logits=logits,
            hidden_states=bert_outputs.last_hidden_state,
            attentions=None,
        )
