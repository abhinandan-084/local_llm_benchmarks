#!/bin/bash

# 1. Configuration
MODELS=(
    "Llama-3.2-3B.Q4_K_M.gguf" 
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" 
    "Mistral-Small-24B-Instruct-2501-Q2_K.gguf"
)
MODELS_DIR="../llama.cpp/models"
LLAMA_CLI="../llama.cpp/build/bin/llama-cli"
BENCHMARK_FILE="eu_ai_report.txt"
OUTPUT_CSV="results/llama_cli_results.csv"

# Initialize CSV
echo "model,scenario,tps,ttft_s" > "$OUTPUT_CSV"

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="$MODELS_DIR/$MODEL_NAME"
    if [ ! -f "$MODEL_PATH" ]; then continue; fi

    # # Default NGL for 3B and 8B models. If the model name contains "Mistral" or "24B", drop NGL to 24 for VRAM safety
    BASE_NGL=80
    [[ "$MODEL_NAME" == *"Mistral"* || "$MODEL_NAME" == *"24B"* ]] && BASE_NGL=24

    for S_TYPE in "Simple_QA" "Coding_Logic" "Long_Context"; do
        echo "Running $MODEL_NAME | $S_TYPE..."

        # Logic for Scenario Params
        if [ "$S_TYPE" == "Simple_QA" ]; then
            P_TOKENS=128   # Estimated prompt tokens
            ARGS=(-p "Explain the difference and relation between QKV and KV Cache." -c 2048)
        elif [ "$S_TYPE" == "Coding_Logic" ]; then
            P_TOKENS=512   # Estimated prompt tokens
            ARGS=(-p "Write a Python script for Tiled Matrix Multiplication using NumPy and Numba. Explain cache locality." -c 2048)
        else
            P_TOKENS=16384
            BASE_NGL=18
            ARGS=(-f "$BENCHMARK_FILE" -p "Act as a specialized Legal Compliance Officer. Using the provided text from Chapter 4 of the EU AI Act (Transparency Obligations), create a comprehensive summary and focus on interaction transparency, sensitive use cases and exceptions." -c 16384 --cache-type-k q4_0 --cache-type-v q4_0)
        fi

        # LLAMA-CLI Arguments:
        "$LLAMA_CLI" -m "$MODEL_PATH" "${ARGS[@]}" \
            -n 512 -ngl $BASE_NGL -fa 1 --temp 0 \
            --simple-io --no-display-prompt > raw_output.tmp 2>&1

        # Extract and clean output
        CLEAN_DATA=$(sed 's/\x1B\[[0-9;]*[JKmsu]//g' raw_output.tmp)

        # Extract Generation TPS
        # Targets "Generation: 7.1 t/s" OR "eval time = ... ( 7.1 tokens per second)"
        TPS=$(echo "$CLEAN_DATA" | grep -oE "Generation: [0-9.]+" | awk '{print $2}')
        if [ -z "$TPS" ]; then
            TPS=$(echo "$CLEAN_DATA" | grep "eval time =" | tail -n 1 | awk -F'(' '{print $2}' | awk '{print $1}')
        fi

        # Extract Prompt TPS and Calculate TTFT
        P_TPS=$(echo "$CLEAN_DATA" | grep -oE "Prompt: [0-9.]+" | awk '{print $2}')
        if [ -z "$P_TPS" ]; then
            # Fallback to ms-based timing if the bracket format isn't used
            TTFT_MS=$(echo "$CLEAN_DATA" | grep "prompt eval time =" | awk '{print $6}')
            if [ ! -z "$TTFT_MS" ]; then
                TTFT=$(awk "BEGIN {print $TTFT_MS / 1000}")
            else
                TTFT="0.00"
            fi
        else
            # TTFT = Tokens / (Tokens/Sec)
            TTFT=$(awk "BEGIN {print $P_TOKENS / $P_TPS}")
        fi

        # Final cleaning
        [[ -z "$TPS" ]] && TPS="0.00"
        [[ -z "$TTFT" ]] && TTFT="0.00"

        # Record and show results
        echo "$MODEL_NAME,$S_TYPE,$TPS,$TTFT" >> "$OUTPUT_CSV"
        echo "   📊 Result: $TPS t/s | $TTFT s"
        
        # Remove temp file
        rm -f raw_output.tmp
    done
done

echo "Llama CLI Test Completed! Results saved to $OUTPUT_CSV"