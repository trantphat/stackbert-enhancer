def generate_kmers(dna_sequence, k):
    # Function to generate k-mers from a given DNA sequence
    kmers = [dna_sequence[i : i + k].upper() for i in range(len(dna_sequence) - k + 1)]
    return kmers


def process_dna_file(file_path, label, k):
    # Open and read the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    results = []

    for i in range(0, len(lines), 2):  # Update the step to 2
        dna_name = lines[i].strip()
        dna_sequence = lines[i + 1].strip()  # Take the next line as the DNA sequence

        kmers = generate_kmers(dna_sequence, k)
        kmer_string = " ".join(kmers) + "\t" + str(label)  # Split DNA sequence and label by a tab
        results.append(kmer_string)

    return results


def write_results_to_file(results, output_file_path):
    with open(output_file_path, "w") as file:
        for i, result in enumerate(results):
            if i < len(results) - 1:
                file.write(result + "\n")
            else:
                file.write(result)


# Parameters
n = 1484  # Change this value to 100 when working with the test files
k_values = range(3, 7)  # k values from 3 to 6

# Define file paths
enhancer_file_paths = {200: "enhancer_200_test.txt", 1484: "enhancer_1484_train.txt"}
non_enhancer_file_paths = {200: "non_enhancer_200_test.txt", 1484: "non_enhancer_1484_train.txt"}

# Check for valid n value
if n not in enhancer_file_paths or n not in non_enhancer_file_paths:
    raise ValueError("Invalid n value. n should be either 200 or 1484.")

# Loop through k values and process files
for k in k_values:
    # Determine output file path based on n value
    if n == 1484:
        output_file_path = f"{k}-mer_identification_train.txt"  # Training file
    elif n == 200:
        output_file_path = f"{k}-mer_identification_test.txt"  # Test file

    # Process files and write results
    enhancer_file_path = enhancer_file_paths[n]
    non_enhancer_file_path = non_enhancer_file_paths[n]

    kmer_results = []
    kmer_results.extend(process_dna_file(non_enhancer_file_path, label=0, k=k))  # Process non enhancer file with label 0
    kmer_results.extend(process_dna_file(enhancer_file_path, label=1, k=k))  # Process enhancer file with label 1

    write_results_to_file(kmer_results, output_file_path)

    print(f"Results written to {output_file_path} for k={k}")