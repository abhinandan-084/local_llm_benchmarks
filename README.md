# local_llm_benchmarks

Reproducible benchmarks that measure the real cost of running local LLMs through a high-level wrapper (Ollama) versus bare-metal `llama.cpp` on constrained, single-GPU hardware. If you run inference on a 6GB consumer card and your tokens-per-second falls off a cliff the moment your context window grows, this repo quantifies exactly why — and shows the tuning flags that buy back up to ~2x throughput. Every number here was produced by automated sweeps on an Intel i5-13450HX + NVIDIA RTX 3050 (6GB VRAM), and all scripts, configs, and result sets are included so you can rerun them on your own box.

## Badges

![Build](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Engine](https://img.shields.io/badge/engine-llama.cpp%20%7C%20Ollama-orange)

## Demo

The headline result: on an 8B model pushed to the edge of a 6GB VRAM budget, switching from Ollama to a manually tuned `llama.cpp` build nearly doubles decode throughput in long-context workloads.

![Long-context throughput by engine](assets/engine_comparison.png)

Ollama conservatively spills KV-cache layers into system RAM to avoid an OOM crash; `llama.cpp`, told explicitly how many layers to keep resident, holds them in VRAM and avoids the PCIe penalty.

## Key Features

- **Head-to-head engine benchmarks.** Identical prompts and models run through both Ollama and a CUDA-compiled `llama.cpp` build, isolating the orchestration overhead ("abstraction tax").
- **Three model size classes.** Coverage of a model that fits entirely in VRAM (Llama 3.2 3B), one that sits right on the 6GB boundary (Llama 3.1 8B), and one that vastly overflows it (Mistral 24B), so you can see where the tradeoffs flip.
- **Three real workload shapes.** Short QA (128-token prompt), coding logic (512-token prompt), and long-context summarization (16,384-token prompt) — covering both compute-bound prefill and memory-bandwidth-bound decode.
- **Low-level tuning sweeps.** Automated `llama-bench` runs across CPU thread counts (`-t`), KV-cache quantization (`-ctk`/`-ctv`), GPU layer offload (`-ngl`), and batch / micro-batch sizing (`-b`/`-ub`).
- **Fully reproducible.** Bash drivers, Python analysis, raw CSV outputs, and the exact flags used for every data point.

## Architecture

Each user request to a local LLM passes through two hardware-bound phases. Knowing which one is your bottleneck dictates which flags actually matter.

![LLM Phases : Prefill and decode](assets/prefill_decode.png)

- **Prefill** ingests the whole prompt in parallel and saturates the GPU's tensor cores. Throughput here scales with raw compute and with how large a micro-batch you can feed the GPU.
- **Decode** generates one token at a time, fetching model weights from VRAM on every step. Tokens-per-second is governed by memory bandwidth, not FLOPs.
- The **KV cache** stores the Key/Value tensors of all prior tokens so the model never recomputes history, turning per-token cost from `O(N²)` down to linear. Its footprint grows linearly with context length:

  ```text
  KV cache size = 2 (K & V) × L (layers) × H_kv (KV heads) × D_head × N (context) × B (bytes/param)
  ```

  For Llama 3.1 8B (Q4_K_M) at a 16k context in FP16, the cache alone is ~2.15 GB on top of ~4.50 GB of weights — ~6.65 GB total, which is the wall a 6GB card hits.

## Installation

Requires an NVIDIA GPU with a working CUDA toolkit, Python 3.10+, and a C/C++ build toolchain.

```bash
git clone https://github.com/abhinandan-084/local_llm_benchmarks.git
cd local_llm_benchmarks

# Python deps for parsing and plotting results
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Build `llama.cpp` natively with CUDA support — this is the bare-metal path under test:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

Install Ollama (the wrapper path under test) from the official installer for your platform, then pull the models:

```bash
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull mistral:24b
```

For `llama.cpp`, place the equivalent GGUF quantizations (e.g. `Q4_K_M`) under `models/`.

## Quickstart

Got everything installed? Reproduce the headline result and the charts above in three commands:

```bash
# 1. Run the engine comparison (all models × scenarios) → results/engine_comparison.csv
python benchmark.py

# 2. Run the four low-level tuning sweeps with llama-bench
./llama_bench.sh
```

## Usage Examples

Run the full engine comparison across all models and scenarios:

```bash
./llama_cli.sh
```

Reproduce a single tuning sweep with `llama-bench`. 

**CPU threads** — match `-t` to *physical* cores, not logical threads:

```bash
./build/bin/llama-bench \
  -m models/mistral-24b-Q4_K_M.gguf \
  -t 4,6,8,10,12 \
  -p 512
```

**KV-cache quantization** — keep Key and Value precision symmetric to stay on the optimized kernels:

```bash
./build/bin/llama-cli \
  -m models/llama-3.1-8b-Q4_K_M.gguf \
  -c 16384 -ctk q4_0 -ctv q4_0 \
  -ngl 33 -p "$(cat prompts/long_context_16k.txt)"
```

**Micro-batch sizing** — the biggest lever for prefill-heavy (RAG) pipelines:

```bash
./build/bin/llama-bench \
  -m models/llama-3.1-8b-Q4_K_M.gguf \
  -b 512,2048 -ub 128,512 \
  -p 16384
```

Parse raw output and regenerate the comparison tables and charts:

```bash
python analysis/summarize.py results/engine_comparison.csv
```

## Results / Benchmarks

All figures are tokens-per-second on the RTX 3050 (6GB) + i5-13450HX rig.

### Engine comparison — llama.cpp vs Ollama

| Scenario | Model | llama.cpp | Ollama | Uplift |
|---|---|---:|---:|---:|
| Simple QA | Llama 3.2 3B | 73.1 | 69.5 | +5.1% |
| Simple QA | Llama 3.1 8B | 34.3 | 32.6 | +5.3% |
| Simple QA | Mistral 24B | 8.6 | 6.7 | +28.0% |
| Coding Logic | Llama 3.2 3B | 71.5 | 68.7 | +4.2% |
| Coding Logic | Llama 3.1 8B | 33.1 | 33.0 | +0.3% |
| Coding Logic | Mistral 24B | 8.5 | 6.7 | +27.4% |
| Long Context | Llama 3.2 3B | 32.8 | 38.9 | −15.8% |
| Long Context | Llama 3.1 8B | 15.3 | 7.7 | +99.5% |
| Long Context | Mistral 24B | 6.6 | 2.9 | +125.3% |

Takeaways: when the model comfortably fits in VRAM (3B), the wrapper is well optimized and can even win on long context. The gap appears precisely where memory pressure is highest — an 8B model on the VRAM boundary, or an oversized 24B model thrashing the PCIe bus. The heavier the bottleneck, the more bare-metal tuning pays off.

### CPU thread tuning (`-t`) — prefill speed

The i5-13450HX has 6 P-cores + 4 E-cores (10 physical). Exceeding the physical core count stalls fast P-cores while they wait on slower E-cores.

| Threads | Prefill (tok/s) |
|---:|---:|
| 4 | 224.3 |
| 6 | 214.4 |
| 8 | 222.7 |
| **10** | **263.9** |
| 12 | 217.9 |

![Prefill speed vs thread count](assets/thread_sweep.png)

→ Set `-t` to your physical core count; ignore hyper-threading.
[Sweep details](https://gist.github.com/abhinandan-084/e6ea4d56b5a5582a2b926d5a7cd5d443)

### KV-cache quantization (`-ctk` / `-ctv`)

Symmetric precision routes through optimized kernels; mismatched K/V precision forces mid-execution format conversion and collapses prefill.

| KV cache (K/V) | Prefill (tok/s) | Decode (tok/s) |
|---|---:|---:|
| q8_0 / q8_0 | 885.3 | 19.3 |
| q4_0 / q4_0 | 862.6 | 20.3 |
| f16 / f16 | 678.3 | 17.7 |
| q4_0 / f16 (mixed) | 55.0 | 16.8 |
| f16 / q8_0 (mixed) | 33.4 | 16.1 |

![Prefill and decode vs KV-cache precision](assets/kv_cache_quant.png)

→ Keep K and V on the same format. Symmetric `q4_0` lifts decode from 17.7 → 20.3 tok/s and cuts the cache footprint by ~75%.
[Sweep details](https://gist.github.com/abhinandan-084/0eada43c2ee70dd13d3bf1ca9a8fcff3)

### GPU layer offload (`-ngl`) — decode speed

Light offloading is a trap: moving intermediate tensors over PCIe costs more than the GPU saves until you cross ~50% of layers.

| Layers offloaded | Decode (tok/s) |
|---:|---:|
| 0 | 10.35 |
| 4 | 9.40 |
| 8 | 9.63 |
| 16 | 13.66 |
| 24 | 20.47 |
| 32 | 30.03 |

![Decode speed vs layers offloaded](assets/offload_cliff.png)

→ Offload (almost) all layers or (almost) none. A half-loaded model is the worst case.
[Sweep details](https://gist.github.com/abhinandan-084/425f8c4c09100b912badd03ce551464a)

### Batch vs. micro-batch (`-b` / `-ub`) — prefill speed

Raising the micro-batch from 128 → 512 sharply increases prefill (at the cost of more VRAM headroom). Decode is unaffected — making this the single biggest lever for prefill-heavy RAG workloads.

| n_batch | ub=128 | ub=512 | Uplift |
|---:|---:|---:|---:|
| 512 | 482 | 893 | +85.3% |
| 2048 | 619 | 877 | +41.7% |

[Sweep details](https://gist.github.com/abhinandan-084/16b4febd1a8dc7c45a8f619355cc0cbf)

## Contributing

Benchmark data from other hardware is the most valuable contribution this repo can get — the whole point is to map how the tradeoffs shift across GPUs, VRAM budgets, and CPU topologies. To submit results:

1. **Fork and branch.** Create a branch named for your rig, e.g. `results/rtx4060-8gb`.
2. **Run the suite unmodified.** Use the committed prompts, models, and flags so numbers stay comparable. Don't change `prompts/` or sweep ranges in a results PR.
3. **Record your environment.** Add a `results/<your-rig>/env.md` capturing GPU, VRAM, CPU (physical P/E core counts), CUDA version, driver, and the `llama.cpp` commit you built.
4. **Commit the raw CSVs**, not just summaries.
5. **Open a PR** describing the hardware and anything notable (OOM thresholds, where the offload cliff landed for you).

Found a methodology issue or a flag that skews comparability? Open an issue first so we can discuss before the data diverges.

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

---

Benchmarks and analysis by **Abhinandan Malhotra**, Senior Data Scientist (London).
