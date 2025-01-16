import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        """
        Initializes the dataset with encodings and labels.

        Args:
            encodings (dict): A dictionary where the keys are feature names and the values are lists of feature values.
            labels (list): A list of labels corresponding to each sample.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Retrieves the sample at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the encodings and the corresponding label for the sample.
        """
        # Convert encodings to tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add label to the item dictionary
        if isinstance(self.labels[idx], str):
            item["labels"] = self.labels[idx]
        else:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.labels)


def load_dataset(args, validation=False, eplison=0):
    """
    Loads and preprocesses a dataset for text classification, with optional label smoothing.

    Args:
        args (Namespace): An object containing the following attributes:
            - train_data_path (str): Path to the training data file.
            - test_data_path (str): Path to the test/validation data file.
            - model_path (str): Path to the pre-trained model for tokenization.
            - max_length (int): Maximum sequence length for tokenization.
        validation (bool, optional): Whether to load the validation dataset. Defaults to `False`.
        eplison (float, optional): The smoothing factor for label smoothing. Defaults to `0`.

    Returns:
        TextClassificationDataset: A dataset object containing tokenized text and corresponding labels.
    """
    if validation:
        data_file_path = args.test_data_path
        training = False
    else:
        data_file_path = args.train_data_path
        training = True

    # Load the dataset from the specified file
    df_raw = pd.read_csv(data_file_path, sep="\t", header=None, names=["text", "label"])

    # Apply label smoothing if in training mode and epsilon is greater than 0
    if training and eplison > 0:
        dataset_label = label_smoothing(list(df_raw["label"]), eplison, training)
    else:
        dataset_label = list(df_raw["label"])

    dataset_text = list(df_raw["text"])

    # Tokenization
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    dataset_encoding = tokenizer(dataset_text, truncation=True, padding=True, max_length=args.max_length)

    # Create the dataset object
    dataset = TextClassificationDataset(dataset_encoding, dataset_label)

    return dataset


def label_smoothing(data_labels, eplison, training):
    """
    Applies label smoothing to the provided labels during training.

    Label smoothing is a regularization technique that adjusts "hard" class labels
    to "soft" labels, reducing overconfidence in predictions and improving model generalization.

    Args:
        data_labels (list): A list of original labels (0 or 1) for the dataset.
        eplison (float): The smoothing factor. Determines how much to adjust the labels.
        training (bool): Whether the function is being applied during training.
                         Label smoothing is only applied if this is `True`.

    Returns:
        list: A list of smoothed labels. Labels are adjusted as follows:
              - If the label is 0, it is replaced with `eplison`.
              - If the label is 1, it is replaced with `1 - eplison`.
              If `training` is `False`, the original labels are returned unchanged.
    """
    if training:
        length = len(data_labels)  # Length of the labels list
        for index in range(length):
            if data_labels[index] == 0:
                data_labels[index] = eplison  # Apply smoothing to label 0
            else:
                data_labels[index] = 1 - eplison  # Apply smoothing to label 1
    return data_labels
