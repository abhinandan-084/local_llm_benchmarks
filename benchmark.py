import requests
import time
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Model configs
model_list = ["llama3.2:3b", "llama3.1:8b-instruct-q4_K_M","hf.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF:Q2_K"]
report_txt = "eu_ai_report.txt" 
ollama_url = "http://localhost:11434/api/generate"

def load_context(file_path):
    if not os.path.exists(file_path):
        # Create a dummy file if it doesn't exist for testing
        with open(file_path, "w") as f:
            f.write("Dummy text for benchmarking content..." * 1000)
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
def run_benchmark(model, scenario, prompt):
    # Context size setting to match llama-bench -p value (e.g., 16384 for Long Context)
    context_size = 16384 if scenario == "Long_Context" else 2048
    print(f"Running {scenario} test on {model}...")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": context_size,      # This is the equivalent of -p in llama-bench
            "num_predict": 512,           # This is the equivalent of -n in llama-bench
            "temperature": 0              # Set to 0 for deterministic benchmarking
            } 
    }
    
    start = time.time()
    try:
        response = requests.post(ollama_url, json=payload).json()
        # Duration is in nanoseconds from Ollama API
        tps = response.get('eval_count', 0) / (response.get('eval_duration', 1) / 1e9)
        ttft = response.get('prompt_eval_duration', 0) / 1e9
        return {"model": model, "scenario": scenario, "tps": round(tps, 2), "ttft": round(ttft, 3)}
    except Exception as e:
        print(f"Error testing {model}: {e}")
        return None
    
def main():
    results = []
    context_text = load_context(report_txt)

    scenarios = {
        "Simple_QKV": "Explain the difference and relation between QKV and KV Cache.",
        "Coding_Logic": "Write a Python script for Tiled Matrix Multiplication using NumPy and Numba. Explain cache locality.",
        "Long_Context": f"{context_text}\n\nAct as a specialized Legal Compliance Officer. Using the provided text from Chapter 4 of the EU AI Act (Transparency Obligations), create a comprehensive summary and focus on interaction transparency, sensitive use cases and exceptions."
    }

    for model in model_list:
        for name, prompt in scenarios.items():
            res = run_benchmark(model, name, prompt)
            if res: results.append(res)

    # Save to JSON for llama.cpp comparison later
    with open("ollama_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nOllama Benchmarks Complete. Results saved to ollama_results.json")

if __name__ == "__main__":
    main()