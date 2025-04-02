import numpy as np
import torch


def compute_kmer_attention_vectors(attention_scores, kmer):
    """
    Compute k-mer aggregated attention vectors from the attention scores.

    Args:
        attention_scores (numpy.ndarray): Attention scores matrix of shape
                                           (num_sequences, num_heads, sequence_length, sequence_length).
        kmer (int): Size of k-mer to aggregate attention scores.
        
    Returns:
        numpy.ndarray: Aggregated attention vectors for each sequence.
    """
    num_sequences, _, sequence_length, _ = attention_scores.shape
    print(f"Attention scores shape for {kmer}-mer: {attention_scores.shape}")

    # Convert token count (with [CLS] and [SEP]) back to original sequence length (L): L = tokens - 2 + k - 1
    original_sequence_length = sequence_length - 1 - 2 + kmer  # original_sequence_length should be fixed at 200

    # Pre-allocate output array
    kmer_attention_vectors = np.zeros((num_sequences, original_sequence_length))  # (num_sequences, original_sequence_length)

    # Iterate over each sequence's attention scores and compute k-mer aggregated attention scores
    for sequence_idx, sequence_attention in enumerate(attention_scores):
        # sequence_attention shape: (num_heads, sequence_length, sequence_length)

        # Step 1: Slice relevant portion of the attention scores (ignoring [CLS] and [SEP])
        relevant_attention = sequence_attention[:, 0, 1 : sequence_length - 1]  # (num_heads, sequence_length - 2)

        # Step 2: Sum across all attention heads for each position
        nucleotide_attention_scores = np.sum(relevant_attention, axis=0)  #  (sequence_length - 2)

        # Step 3: Convert the result to float
        nucleotide_attention_scores = nucleotide_attention_scores.astype(float)

        # Aggregate attention scores across k-mer positions
        counts = np.zeros(original_sequence_length)
        aggregated_scores = np.zeros(original_sequence_length)

        # Iterate over each position in the nucleotide attention scores
        for position in range(len(nucleotide_attention_scores)):
            # Get the attention score at the current position
            score = nucleotide_attention_scores[position]

            # Distribute this score across the next k-mer positions
            for offset in range(kmer):
                # Update counts and aggregated scores for the corresponding position
                counts[position + offset] += 1.0  # Increment the count
                aggregated_scores[position + offset] += score  # Add the score

        # Average the aggregated scores by dividing by the counts
        aggregated_scores = aggregated_scores / counts
        # Normalize the aggregated scores to have an L2 Euclidean unit norm
        aggregated_scores = aggregated_scores / np.linalg.norm(aggregated_scores)

        # Store the normalized k-mer attention vector for the current sequence
        kmer_attention_vectors[sequence_idx] = aggregated_scores

    return kmer_attention_vectors


def extract_kmer_attention_vectors(model, dataloader, config):
    """
    Extract and compute k-mer aggregated attention vectors from a model and dataloader.

    Args:
        model (torch.nn.Module): The model from which to extract attention scores.
        dataloader (torch.utils.data.DataLoader): Dataloader providing input data.
        config (object): Configuration object containing settings such as k-mer size and batch size.

    Returns:
        numpy.ndarray: Aggregated attention vectors for all sequences in the dataset.
    """
    # Extract the k-mer size from the configuration
    kmer = config.kmer

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize an array to store attention scores for all samples
    # Shape: (total_samples, num_heads, max_seq_length, max_seq_length)
    total_samples = len(dataloader.dataset)
    attention_scores_matrix = np.zeros([total_samples, 12, config.max_length - kmer + 3, config.max_length - kmer + 3])

    # Iterate through batches in the dataloader
    for batch_idx, batch_data in enumerate(dataloader):
        model.eval()
        with torch.no_grad():
            # Move input data to the appropriate device (CPU or GPU)
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)

            # Get model's attention scores from the last layer
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract last layer's attention scores
        last_layer_attention = outputs[-1]  # (batch_size, num_heads, sequence_length, sequence_length)

        # Save the batch's attention scores into the overall matrix
        start_idx = batch_idx * config.batch_size
        end_idx = start_idx + len(batch_data["input_ids"])
        attention_scores_matrix[start_idx:end_idx] = last_layer_attention.cpu().numpy()

    # Compute final k-mer aggregated attention vectors using the computed scores
    kmer_attention_vectors = compute_kmer_attention_vectors(attention_scores_matrix, kmer)  # (total_samples, sequence_length)

    return kmer_attention_vectors
