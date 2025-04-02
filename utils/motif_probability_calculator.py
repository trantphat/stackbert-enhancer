import os
import re
import numpy as np
import pandas as pd


def calculate_probabilities(seq_dict):
    """
    Calculate the probability matrix for nucleotide occurrences.

    Args:
        seq_dict (dict): A dictionary where each key is an index and the value is a dictionary 
                         with nucleotide counts (e.g., {"A": 0, "C": 0, "G": 0, "T": 0}).

    Returns:
        list: A list of tuples containing probabilities for A, C, G, T at each position.
    """
    probabilities = []
    for key, value in seq_dict.items():
        count_A = value["A"]
        count_C = value["C"]
        count_G = value["G"]
        count_T = value["T"]

        total_count = count_A + count_C + count_G + count_T

        if total_count > 0:
            prob_A = count_A / total_count
            prob_C = count_C / total_count
            prob_G = count_G / total_count
            prob_T = count_T / total_count
        else:
            prob_A = prob_C = prob_G = prob_T = 0

        probabilities.append((prob_A, prob_C, prob_G, prob_T))
    return probabilities


def calculate_sequence_probabilities(sequences):
    """
    Calculate nucleotide probabilities for each position in a set of sequences.

    Args:
        sequences (list): List of DNA sequences.

    Returns:
        list: A list of tuples containing probabilities for A, C, G, T at each position.
    """
    position_counts = {}
    for seq in sequences:
        for idx, nucleotide in enumerate(seq.upper()):
            if idx not in position_counts:
                position_counts[idx] = {"A": 0, "C": 0, "G": 0, "T": 0}
            if nucleotide in position_counts[idx]:
                position_counts[idx][nucleotide] += 1

    return calculate_probabilities(position_counts)


def save_tomtom_format(probabilities, file_name, motif_name="motif"):
    """
    Save nucleotide probability matrix in TomTom-compatible format.

    Args:
        probabilities (list): List of tuples with nucleotide probabilities.
        file_name (str): Output file name.
        motif_name (str): Name of the motif.
    """
    width = len(probabilities)
    content = (
        f"MEME version 4\n"
        f"ALPHABET=ACGT\n"
        f"strands:+ -\n"
        f"MOTIF {motif_name}\n"
        f"letter-probability matrix: alength=4 w={width}\n"
    )

    for probs in probabilities:
        content += "  ".join(f"{p:.4f}" for p in probs) + "\n"

    with open(file_name, "w") as file:
        file.write(content)


def load_sequences(file_path, has_label=True):
    """
    Load DNA sequences from a file.

    Args:
        file_path (str): Path to the input file.
        has_label (bool): Whether the file contains labels.

    Returns:
        tuple or np.ndarray: If `has_label` is True, returns positive and negative sequences.
                             Otherwise, returns all sequences.
    """
    df = pd.read_csv(file_path, sep="\t", header=None)
    
    if has_label:
        df.columns = ["sequence", "label"]
        sequences = df["sequence"].values
        labels = df["label"].values

        pos_sequences = sequences[labels == 1]
        neg_sequences = sequences[labels == 0]

        return pos_sequences, neg_sequences
    else:
        df.columns = ["sequence"]
        return df["sequence"].values


def process_motif_files(input_dir):
    """
    Process all motif files in a directory and save them in TomTom format.

    Args:
        input_dir (str): Directory containing input files.
    """
    for file_name in os.listdir(input_dir):
        if re.match(r".*\.txt$", file_name):
            full_path = os.path.join(input_dir, file_name)
            sequences = load_sequences(full_path, has_label=False)
            probabilities = calculate_sequence_probabilities(sequences)

            output_file_name = os.path.join(
                input_dir,
                f"{os.path.splitext(file_name)[0]}_TomTom.txt"
            )
            motif_name = os.path.splitext(file_name)[0].split("_")[-1]
            save_tomtom_format(probabilities, output_file_name, motif_name)


def combine_tomtom_files(input_dir, output_file_name):
    """
    Combine all TomTom-formatted files into a single file.

    Args:
        input_dir (str): Directory containing TomTom-formatted files.
        output_file_name (str): Name of the combined output file.
    """
    combined_content = ""
    
    for index, file_name in enumerate(os.listdir(input_dir)):
        if re.match(r".*_TomTom\.txt$", file_name):
            full_path = os.path.join(input_dir, file_name)
            
            with open(full_path, "r") as file:
                lines = file.readlines()
                
                if index == 0:  # Include header from the first file
                    combined_content += "".join(lines)
                else:  # Skip header lines in subsequent files
                    combined_content += "".join(lines[3:])
            
            combined_content += "\n"

    with open(os.path.join(input_dir, output_file_name), "w") as output_file:
        output_file.write(combined_content)