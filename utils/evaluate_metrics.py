import math
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def evaluate_metrics(logits, true_labels):
    """
    Evaluate performance metrics for binary classification.

    Args:
        logits (array-like): Predicted logits or probabilities from the model.
        true_labels (array-like): True labels corresponding to the input data.

    Returns:
        tuple: A tuple containing the following metrics:
            - accuracy (float): The proportion of correctly classified samples.
            - sensitivity (float): Also known as recall or true positive rate; measures the ability to correctly identify positive samples.
            - specificity (float): Measures the ability to correctly identify negative samples.
            - mcc (float): Matthews correlation coefficient; a balanced measure of classification quality, even for imbalanced datasets.
            - auc (float): Area under the ROC curve; evaluates the model's ability to distinguish between classes.
    """
    # Convert logits and labels to NumPy arrays for efficient processing
    logits = np.array(logits)
    true_labels = np.array(true_labels)

    threshold = 0.55

    # Create predicted labels based on the threshold
    predicted_labels = [1 if logit > threshold else 0 for logit in logits]

    # Create true labels based on the threshold
    true_binary_labels = [1 if label > threshold else 0 for label in true_labels]

    # Calculate confusion matrix
    TN, FP, FN, TP = confusion_matrix(true_binary_labels, predicted_labels).ravel()

    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    mcc = (TP * TN - FP * FN) / math.sqrt((TP + FN) * (TP + FP) * (FP + TN) * (FN + TN)) if (TP + FN) * (TP + FP) * (FP + TN) * (FN + TN) > 0 else 0
    auc = roc_auc_score(true_binary_labels, logits) if len(np.unique(true_binary_labels)) > 1 else 0

    return accuracy, sensitivity, specificity, mcc, auc
