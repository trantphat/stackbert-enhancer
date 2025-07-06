# Define a function to process the input files and extract DNA sequences with labels
def process_file(input_file, label):
    sequences = []
    try:
        with open(input_file, 'r') as file:
            lines = file.readlines()
            for i in range(1, len(lines), 2):  # DNA sequences are on every second line
                sequence = lines[i].strip().upper()  # Convert to uppercase
                sequences.append(f"{sequence}\t{label}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
    return sequences

# Process both enhancer and non-enhancer files
non_enhancer_sequences = process_file('non_enhancer_1484_train.txt', '0')
enhancer_sequences = process_file('enhancer_1484_train.txt', '1')

# Combine the sequences
all_sequences =  non_enhancer_sequences + enhancer_sequences

# Write the combined sequences to the output file
output_file = 'combined_dna_sequences.txt'
try:
    with open(output_file, 'w') as file:
        for sequence in all_sequences:
            file.write(sequence + '\n')
    print(f"Data successfully written to {output_file}.")
except Exception as e:
    print(f"Error writing to {output_file}: {e}")
