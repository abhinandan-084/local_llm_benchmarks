# **Beyond Ollama: Squeezing a 24B Model into a 6GB "Workhorse"**

This repository contains benchmarking scripts, raw data, and optimization configurations comparing Ollama vs. llama.cpp (llama-cli).

I'm running these tests on an old gaming laptop equipped with an RTX 3050 6GB. For a long time, Ollama was my go-to—the "Apple of Local LLMs." But as I started pushing for larger models (like Mistral 24B), I hit the "Ollama Box." I felt restricted by the lack of granular control over VRAM offloading and the inability to tweak parameters for limited hardware.

This project documents how going "bare metal" with llama.cpp can turn aging hardware into a capable AI workhorse.

---

# **Base Benchmark: Ollama vs. llama.cpp**

I tested three models across three real-world scenarios:
1. **Llama 3.2 3B**: Fully fits in GPU VRAM.
2. **Llama 3.1 8B**: Fits in GPU with minimal space left for KV cache.
3. **Mistral 24B**: Massive model; forces Ollama to offload mostly to CPU.

### **Performance Comparison (Tokens Per Second)**

| Model | Scenario | Ollama (t/s) | llama.cpp (t/s) | **Performance Uplift** |
| :--- | :--- | :--- | :--- | :--- |
| **Llama 3.2 3B** | Simple Q&A | 50.6 | 62.7 | **+24%** |
| **Llama 3.1 8B** | Coding Logic | 15.4 | 30.9 | **+101%** |
| **Mistral 24B** | Long Context | 2.6 | 2.8 | **+7%** |

> **Note on the 8B Model:** The +101% uplift in `llama-bench` represents the theoretical peak efficiency of CUDA kernels when perfectly saturated. In real-world `llama-cli` usage, the gain is more modest (~15-20%) but results in a significantly snappier and more responsive experience.

### **The "TTFT" Mystery**
In my results, Ollama sometimes claims a near-instant **Time to First Token (TTFT)**. **Don't be fooled.** Ollama's logs often measure the time to the *start* of the server response. `llama-cli` gives you the raw reality of the **Prompt Processing (Prefill)** stage. When feeding 16k context into a 6GB GPU, that "real" wait time is where the hardware is actually doing the heavy lifting.

---

# **Advanced Benchmarks : Optimising LLama.cpp Performance**

Proving Llama.cpp is faster was the first step. The real engineering happens when we isolate specific hardware bottlenecks. For the advanced profiling, use the following `llama-bench` commands.

### **KV Cache Quantization Trade-offs (`-ctk` / `-ctv`)**
When processing long contexts (like 16k tokens), the Key-Value (KV) cache balloons rapidly. On an 8B model, a 16k uncompressed cache takes up ~2.15 GB of VRAM. If this spills over the 6GB limit, the system crashes. We can compress this cache dynamically to save VRAM.

**The Test:** Testing for cache precision from 16-bit float down to 4-bit quantized.

```bash
./build/bin/llama-bench \
  -m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -ngl 24 \
  -fa 1 \
  -p 4096 \
  -n 128 \
  -ctk f16,q8_0,q4_0 \
  -ctv f16,q8_0,q4_0 \
  -o csv > kv_cache_tradeoff.csv
```

**Finding:** Mixed precision (e.g., f16 Keys and q4_0 Values) destroys tensor core efficiency. However, symmetric q4_0/q4_0 quantization reduces VRAM footprint by 75% while actually increasing decode speed due to reduced memory-bus congestion.

### **Batch & Micro-Batch Ingestion (`-b` vs `-ub`)**
When you send a prompt to an AI, the GPU does two completely different types of work. 
1. **Prompt Prefill:** The GPU takes the entire prompt and processes it all at once. Because it knows all the words in the prompt simultaneously, it can use all its thousands of cores to do math in parallel. The bottleneck here is Compute-bound. Because the GPU knows the entire text, it is working at its maximum "thinking" speed (TFLOPS). 
2. **Token Generation:** Once the prompt is read, the model generates the answer one token at a time. To pick the next word, it has to look back at everything it just wrote. The bottleneck thus becomes memory-bound. Even though picking one word isn't "hard", the GPU has to load the entire multi-gigabyte model from its memory (VRAM) into its processor just to calculate that one single word. The model has to do this over and over. The limit is then how fast data can move from storage to the the processor

By adjusting batch sizes, we can tune how the GPU ingests massive blocks of text to prevent the "Warmup Spike" (where transient memory allocations crash the GPU driver).

**The Test:** Altering total batch size vs. physical micro-batch chunks.

```bash
./build/bin/llama-bench \
  -m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -ngl 24 \
  -fa 1 \
  -p 1024 \
  -n 128 \
  -b 512,2048 \
  -ub 128,512 \
  -o csv > batch_size_evaluation.csv
```

**Finding:** Increasing -ub from 128 to 512 massively increases prefill speed (up to +85%), but it requires more available VRAM padding to prevent an OOM crash. Decode speed remains unaffected.

### **The GPU Layer Offloading Sweep (`-ngl`)**
Offloading splits the model's layers between a faster GPU and a slower CPU. Because the model processes information in its layers sequentially, data must constantly jump between these two components to complete a single thought. This transfer happens across the PCIe bus, which is significantly slower than the GPU's internal memory. Every time the data has to use this, it creates a "**latency penalty**" that forces the hardware to pause and wait. Ultimately, the time saved by the GPU's fast math is often canceled out by the time wasted moving data back and forth.

**The Test:** Testing from 0% GPU (pure CPU) to 100% GPU offload.

```bash
./build/bin/llama-bench \
  -m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -fa 1 \
  -p 512 \
  -n 128 \
  -ngl 0,8,16,24,32 \
  -o csv > ngl_test_8b.csv
```

**Finding:** Scaling is non-linear. Offloading just 4 or 8 layers actually slows down token generation compared to pure CPU because the PCIe transfer penalty outweighs the compute gain. Performance only spikes upward once we offload the majority of the model (16+ layers).

### **CPU Thread Topology Optimization (`-t`)**
When a 24B model doesn't fit in 6GB of VRAM, the CPU handles the rest. On an asymmetric processor, sometimes using all available threads often backfires because the slower Efficiency cores (E-cores) create a bottleneck, forcing the fast Performance cores (P-cores) to wait for them to finish.

**The Test:** Testing thread counts to find the asymmetric inflection point for Mistral 24B

```bash
./build/bin/llama-bench \
  -m models/Mistral-Small-24B-Instruct-2501-Q2_K.gguf \
  -ngl 12 \
  -fa 1 \
  -p 512 \
  -n 128 \
  -t 4,6,8,10,12 \
  -o csv > thread_optimization.csv
```

**Finding:** Prefill speed peaks exactly at 10 threads (matching the total physical core count). Pushing to 12 threads forces hyper-threading overhead and E-core context switching, degrading matrix multiplication speeds.

---

## **Installation & Setup**

### **1. Ollama (The Easy Way)**
Perfect for quick testing and general use.
* **Download:** [ollama.com](https://ollama.com)
* **Pulling Models:**
  ```bash
  ollama pull llama3.1:8b-instruct-q4_K_M

### **2. llama.cpp (The Custom Way)**
This is where the control happens. I compiled this with CUDA support to make sure my 3050 was actually being used.
* **build**
  ```bash
  git clone [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
  cd llama.cpp
  cmake -B build -DGGML_CUDA=ON
  cmake --build build --config Release -j

* **Usage: Download a GGUF file and run:**
  ```bash
  ./llama-cli -m models/mistral-24b.gguf -p "Your Prompt" -ngl 12

---

## Lessons Learned (The Hard Way)

### 1. **`llama-cli`** vs. **`llama-bench`**
I used both tools included in the `llama.cpp` build. They share the same engine but serve very different roles:

| Feature | `llama-cli` | `llama-bench` |
| :--- | :--- | :--- |
| **Goal** | Use the model (Inference/Chat) | Measure the model (Profiling) |
| **Output** | Natural Language Text | Performance Data (t/s, ms) |
| **Interaction** | Interactive / Human-centric | Automated / Stress-test |
| **Use Case** | Daily tasks & Coding | Hardware optimization |

> **Summary:** Use **`llama-cli`** when you want to talk to the AI; use **`llama-bench`** to find the absolute speed limit and optimal settings for your hardware.

### 2. **The 6GB VRAM Wall**
Trying to run a 24B model on 6GB of VRAM is like putting a jet engine to a bicycle: you have all the power in the world, but nowhere near enough frame to hold it.

* **The Fix:** Manual **NGL (Number of GPU Layers)** tuning.
* Ollama tries to guess your offload layers, but it often fails on "mid-range" laptop cards. By manually setting `-ngl 12` in `llama-cli`, I kept the system stable and avoided the CUDA "Out of Memory" crashes that plagued my Ollama experience.

### 3. **The "Warmup" Spike**
Small, rapid prompts can sometimes crash a mobile GPU during initialization.

> **Useful Tip:** If you are benchmarking, run a small "throwaway" prompt first to initialize the CUDA context before timing your real tests.

---

## **Conclusion**

If you want convenience, stay with **Ollama**. But if you're running on "scrappy" hardware and want to run models that shouldn't technically fit, **llama.cpp** is the key. It turned my 3050 from a basic GPU into a legitimate AI workhorse.
