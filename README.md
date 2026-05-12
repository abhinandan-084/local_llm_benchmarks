# Beyond Ollama: Squeezing a 24B Model into a 6GB "Workhorse"

This repository contains benchmarking scripts, raw data, and optimization configurations comparing Ollama vs. llama.cpp (llama-cli).

Most AI benchmarks you see online are run on H100s or dual 4090 builds. This isn't one of them.

I’m running these tests on an old gaming laptop equipped with an RTX 3050 6GB. For a long time, Ollama was my go-to—the "Apple of Local LLMs." But as I started pushing for larger models (like Mistral 24B), I hit the "Ollama Box." I felt restricted by the lack of granular control over VRAM offloading and the inability to tweak parameters for limited hardware.

This project documents how going "bare metal" with llama.cpp can turn aging hardware into a capable AI workhorse.

---

## 📊 The Benchmark: Ollama vs. llama.cpp

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

> **⚠️ Note on the 8B Model:** The +101% uplift in `llama-bench` represents the theoretical peak efficiency of CUDA kernels when perfectly saturated. In real-world `llama-cli` usage, the gain is more modest (~15-20%) but results in a significantly snappier and more responsive experience.

### **The "TTFT" Mystery**
In my results, Ollama sometimes claims a near-instant **Time to First Token (TTFT)**. **Don't be fooled.** Ollama's logs often measure the time to the *start* of the server response. `llama-cli` gives you the raw reality of the **Prompt Processing (Prefill)** stage. When feeding 16k context into a 6GB GPU, that "real" wait time is where the hardware is actually doing the heavy lifting.

---

## 🛠️ **Installation & Setup**

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