import torch


def make_predictions(model, dataloader, kmer=3):
    """
    Perform predictions using a given model and dataloader.

    Args:
        model (torch.nn.Module): The model to use for predictions.
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of data.
        kmer (int, optional): An additional parameter passed to the model during the forward pass. Defaults to 3.

    Returns:
        tuple: A tuple containing:
            - predicted_logits (numpy.ndarray): The predicted logits from the model.
            - actual_labels (numpy.ndarray): The ground truth labels corresponding to the input data.
    """
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize lists to store actual labels and predicted logits
    total_iterations = len(dataloader) * 1.0
    actual_labels, predicted_logits = [], []

    for batch in dataloader:
        # Set the model to evaluation mode
        model.eval()
        with torch.no_grad():  # Disable gradient calculation for inference
            # Move input data and labels to the specified device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass through the model with k-mer size
            outputs = model(input_ids, attention_mask, labels, kmer)

        # Extract loss and logits from model outputs
        loss = outputs.loss
        logits = outputs.logits

        # Append or concatenate predicted logits and actual labels
        if len(actual_labels) == 0:
            actual_labels = labels
            predicted_logits = logits
        else:
            actual_labels = torch.cat([actual_labels, labels], dim=0)
            predicted_logits = torch.cat([predicted_logits, logits], dim=0)

    # Convert tensors to numpy arrays
    return predicted_logits.to("cpu").numpy(), actual_labels.to("cpu").numpy()
