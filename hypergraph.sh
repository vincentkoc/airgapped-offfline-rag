#!/bin/bash

# Set the path to your Obsidian vault
HG_PATH="$PWD"

# Function to process files, excluding 'lib' and 'build' directories
process_c_h_files() {
    local dir="$1"
    find "$dir" \( -path "*/.venv/*" -o -path "*/chroma_db/*" \) -prune -o \( -name "*.md" -o -name "*.yml" -o -name "*.yaml" -o -name "*.py" -o -name "*.txt" \) -type f -print | while read -r file; do
        if [ -f "$file" ]; then  # Check if it's a regular file
            echo "File: ${file#$HG_PATH/}"
            echo "---"
            cat "$file"
            echo -e "\n---\n"
        fi
    done
}

# Generate the hypergraph prompt
generate_prompt() {
    echo "You are an AI assistant tasked with analyzing the following 'diet-rag' code repository for a fully offline RAG setup. Each file is separated by '---'. Please process this information and be prepared to answer questions about the code in this repository."
    echo ""
    process_c_h_files "$HG_PATH/"
    # process_c_h_files "$HG_PATH/Friend/firmware/og"
}

# Generate the prompt and copy to clipboard
generate_prompt | pbcopy

echo "Hypergraph prompt has been generated and copied to your clipboard."
