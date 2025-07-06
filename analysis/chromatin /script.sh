for file in *.bigWig; do
    mv "$file" "${file%.bigWig}.bw"
done