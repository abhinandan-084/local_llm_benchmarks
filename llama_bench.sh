#!/bin/bash

# 1. Configuration
MODELS=(
    "Llama-3.2-3B.Q4_K_M.gguf" 
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" 
    "Mistral-Small-24B-Instruct-2501-Q2_K.gguf"
)
MODELS_DIR="../llama.cpp/models/"
GEN_LEN=512 
OUTPUT_CSV="results/llama_bench_results.csv"
FIRST_RUN=true

# Clear previous results
> $OUTPUT_CSV

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="$MODELS_DIR/$MODEL_NAME"
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Skipping $MODEL_NAME: File not found."
        continue
    fi

    # Default NGL for 3B and 8B models
    CURRENT_NGL=80

    # If the model name contains "Mistral" or "24B", drop NGL to 24 for VRAM safety
    if [[ "$MODEL_NAME" == *"Mistral"* || "$MODEL_NAME" == *"24B"* ]]; then
        echo "Large model detected ($MODEL_NAME). Setting NGL to 24 for VRAM safety."
        CURRENT_NGL=24
    fi
    # -------------------------

    # SCENARIO 1: Simple QKV
    echo "Benchmarking $MODEL_NAME | Simple QKV (NGL: $CURRENT_NGL)"
    RAW=$(../llama.cpp/build/bin/llama-bench --output csv -m "$MODEL_PATH" -p 128 -n $GEN_LEN -d 2048 -ngl $CURRENT_NGL -fa 1)

    if [ "$FIRST_RUN" = true ]; then
        # Capture header from the first line of output
        echo "$(echo "$RAW" | head -n 1),scenario" > "$OUTPUT_CSV"
        FIRST_RUN=false
    fi
    echo "$RAW" | tail -n +2 | sed "s/$/,Simple_QKV/" >> "$OUTPUT_CSV"

    # SCENARIO 2: Coding Logic
    echo "Benchmarking $MODEL_NAME | Coding Logic (NGL: $CURRENT_NGL)"
    RAW=$(../llama.cpp/build/bin/llama-bench --output csv -m "$MODEL_PATH" -p 512 -n $GEN_LEN -d 2048 -ngl $CURRENT_NGL -fa 1)
    echo "$RAW" | tail -n +2 | sed "s/$/,Coding_Logic/" >> "$OUTPUT_CSV"

    # SCENARIO 3: Long Context
    echo "Benchmarking $MODEL_NAME | Long Context"
    # For the long context, we need even more VRAM room, so we drop it further for the big model
    SCENARIO_3_NGL=$CURRENT_NGL
    if [ "$CURRENT_NGL" -eq 24 ]; then SCENARIO_3_NGL=18; fi # If it's the big model, use CPU for 16k context

    RAW=$(../llama.cpp/build/bin/llama-bench --output csv -m "$MODEL_PATH" -p 16384 -n $GEN_LEN -d 16384 -ngl $SCENARIO_3_NGL --cache-type-k q4_0 --cache-type-v q4_0 -fa 1)
    echo "$RAW" | tail -n +2 | sed "s/$/,Long_Context/" >> "$OUTPUT_CSV"

done

echo "Llama Bench Completed! Results saved to $OUTPUT_CSV"