def generate_kmers(dna_sequence, k):
    # Function to generate k-mers from a given DNA sequence
    kmers = [dna_sequence[i : i + k].upper() for i in range(len(dna_sequence) - k + 1)]
    return kmers


def process_dna_file(file_path, label, k):
    # Open and read the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    results = []

    for i in range(0, len(lines), 5):
        dna_name = lines[i].strip()
        dna_sequence = "".join(line.strip() for line in lines[i + 1 : i + 5])

        kmers = generate_kmers(dna_sequence, k)
        kmer_string = " ".join(kmers) + "\t" + str(label)  # Split DNA sequence and label by a tab
        results.append(kmer_string)

    return results


def write_results_to_file(results, output_file_path):
    with open(output_file_path, "w") as file:
        for result in results:
            file.write(result + "\n")


# Parameters
n = 742  # Change this value to 100 when working with the test files
k_values = range(3, 7)  # k values from 3 to 6

# Define file paths
strong_file_paths = {100: "strong_100_test.txt", 742: "strong_742_train.txt"}
weak_file_paths = {100: "weak_100_test.txt", 742: "weak_742_train.txt"}

# Check for valid n value
if n not in strong_file_paths or n not in weak_file_paths:
    raise ValueError("Invalid n value. n should be either 100 or 742.")

# Loop through k values and process files
for k in k_values:
    # Determine output file path based on n value
    if n == 742:
        output_file_path = f"{k}-mer_classification_train.txt"  # Training file
    elif n == 100:
        output_file_path = f"{k}-mer_classification_test.txt"  # Test file

    # Process files and write results
    strong_file_path = strong_file_paths[n]
    weak_file_path = weak_file_paths[n]

    kmer_results = []
    kmer_results.extend(process_dna_file(weak_file_path, label=0, k=k))  # Process weak file with label 0
    kmer_results.extend(process_dna_file(strong_file_path, label=1, k=k))  # Process strong file with label 1

    write_results_to_file(kmer_results, output_file_path)

    print(f"Results written to {output_file_path} for k={k}")
