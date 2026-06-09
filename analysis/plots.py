"""Plotting functions for the benchmark suite.

Each function reads a committed result file and writes one figure. The logic is a
faithful port of the original notebooks (notebooks/base_benchmarks.ipynb and
notebooks/advanced_benchmarks.ipynb) — same data handling, same chart types — but
with paths resolved from the repo root so the functions run from anywhere.

Nothing here recomputes a benchmark. The CSVs/JSON are ground truth; these
functions only render them.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never to a GUI window

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# Paths (resolved from repo root, not the current working directory)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
ADVANCED_DIR = REPO_ROOT / "advanced_benchmark_results"
ASSETS_DIR = REPO_ROOT / "assets"

# Map every model identifier that appears in any result file to one display name.
MODEL_MAP = {
    "Llama-3.2-3B.Q4_K_M.gguf": "Llama-3.2-3B",
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf": "Llama-3.1-8B",
    "Mistral-Small-24B-Instruct-2501-Q2_K.gguf": "Mistral-24B",
    "llama3.2:3b": "Llama-3.2-3B",
    "llama3.1:8b-instruct-q4_K_M": "Llama-3.1-8B",
    "hf.co/bartowski/Mistral-Small-24B-Instruct-2501-GGUF:Q2_K": "Mistral-24B",
}


# --------------------------------------------------------------------------- #
# Loaders — turn each raw result file into a tidy (model, scenario, tps, ttft)
# --------------------------------------------------------------------------- #
def _process_llama_bench(csv_path: Path) -> pd.DataFrame:
    """llama-bench emits two rows per config: prompt-processing (n_gen == 0) and
    token-generation (n_prompt == 0). TTFT comes from the former, TPS from the
    latter."""
    df = pd.read_csv(csv_path).replace('"', "", regex=True)
    df["clean_model"] = df["model_filename"].apply(lambda x: str(x).split("/")[-1])
    df["model_name"] = df["clean_model"].map(MODEL_MAP)

    ttft_df = df[df["n_gen"] == 0][["model_name", "scenario", "avg_ns"]].copy()
    ttft_df["ttft"] = ttft_df["avg_ns"] / 1e9

    tps_df = df[df["n_prompt"] == 0][["model_name", "scenario", "avg_ts"]].copy()
    tps_df = tps_df.rename(columns={"avg_ts": "tps"})

    merged = pd.merge(ttft_df, tps_df, on=["model_name", "scenario"])
    merged["benchmark"] = "llama-bench"
    return merged[["model_name", "scenario", "tps", "ttft", "benchmark"]]


def _process_llama_cli(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["model_name"] = df["model"].map(MODEL_MAP)
    df["benchmark"] = "llama-cli"
    return df[["model_name", "scenario", "tps", "ttft_s", "benchmark"]].rename(
        columns={"ttft_s": "ttft"}
    )


def _process_ollama(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["model_name"] = df["model"].map(MODEL_MAP)
    df["benchmark"] = "Ollama"
    return df[["model_name", "scenario", "tps", "ttft", "benchmark"]]


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_engine_comparison(out_path: Path) -> Path:
    """Throughput + responsiveness across engines and models (ports
    base_benchmarks.ipynb). NOTE: this renders the full multi-scenario view your
    notebook produced; review against the currently-published asset before
    overwriting it."""
    df_final = pd.concat(
        [
            _process_llama_bench(RESULTS_DIR / "llama_bench_results.csv"),
            _process_llama_cli(RESULTS_DIR / "llama_cli_results.csv"),
            _process_ollama(RESULTS_DIR / "ollama_results.json"),
        ],
        ignore_index=True,
    )

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    sns.barplot(data=df_final, x="model_name", y="tps", hue="benchmark", ax=ax1, palette="viridis")
    ax1.set_title("Throughput Comparison (Tokens Per Second) - Higher is Better", fontsize=15)
    ax1.set_ylabel("TPS")
    ax1.set_xlabel("")
    ax1.legend(title="Tool", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Log scale: long-context TTFT dwarfs the short scenarios.
    sns.barplot(data=df_final, x="model_name", y="ttft", hue="benchmark", ax=ax2, palette="magma")
    ax2.set_yscale("log")
    ax2.set_title("Responsiveness (Time to First Token) - Lower is Better (Log Scale)", fontsize=15)
    ax2.set_ylabel("Seconds (Log)")
    ax2.set_xlabel("")
    ax2.legend(title="Tool", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_engine_comparison_demo(out_path: Path) -> Path:
    """The headline chart: long-context throughput, tuned llama.cpp vs Ollama
    defaults. Reconstructs the curated demo image from the result files. Colours
    are a close approximation of the hand-made version — adjust to taste."""
    cli = _process_llama_cli(RESULTS_DIR / "llama_cli_results.csv")
    oll = _process_ollama(RESULTS_DIR / "ollama_results.json")

    order = ["Llama-3.2-3B", "Llama-3.1-8B", "Mistral-24B"]
    labels = ["Llama 3.2 3B", "Llama 3.1 8B", "Mistral 24B"]

    def _lc(df: pd.DataFrame) -> list[float]:
        s = df[df["scenario"] == "Long_Context"].set_index("model_name")["tps"]
        return [round(float(s[m]), 1) for m in order]

    cli_vals, oll_vals = _lc(cli), _lc(oll)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = range(len(order))
    width = 0.4
    bars1 = ax.bar([i - width / 2 for i in x], cli_vals, width,
                   label="llama.cpp (tuned)", color="#36a832")
    bars2 = ax.bar([i + width / 2 for i in x], oll_vals, width,
                   label="Ollama (defaults)", color="#6e6e6e")

    ax.set_title("Long-Context (16k) Throughput by Engine", fontsize=15, fontweight="bold")
    ax.set_ylabel("Tokens/s")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(cli_vals + oll_vals) * 1.18)
    ax.legend()
    ax.bar_label(bars1, fmt="%.1f", padding=3)
    ax.bar_label(bars2, fmt="%.1f", padding=3)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _set_advanced_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13, "figure.titlesize": 14}
    )


def plot_batch_size(out_path: Path) -> Path:
    """Prefill vs decode speed across batch / micro-batch sizing."""
    _set_advanced_theme()
    df = pd.read_csv(ADVANCED_DIR / "batch_size_evaluation.csv")
    df["config"] = "Batch:" + df["n_batch"].astype(str) + "\nMicroB:" + df["n_ubatch"].astype(str)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    df_prompt = df[df["n_prompt"] > 0].sort_values("config")
    df_gen = df[df["n_gen"] > 0].sort_values("config")

    sns.barplot(data=df_prompt, hue="config", y="avg_ts", ax=ax1, palette="Blues_r", legend=False)
    ax1.set_title("Prompt Processing Speed (Prefill Phase)")
    ax1.set_ylabel("Tokens per Second")
    ax1.set_xlabel("Engine Sizing Configuration")
    for container in ax1.containers:
        ax1.bar_label(container, fmt="%.1f", padding=3)

    sns.barplot(data=df_gen, hue="config", y="avg_ts", ax=ax2, palette="Oranges_r", legend=False)
    ax2.set_title("Token Generation Speed (Decode Phase)")
    ax2.set_ylabel("Tokens per Second")
    ax2.set_xlabel("Engine Sizing Configuration")
    for container in ax2.containers:
        ax2.bar_label(container, fmt="%.1f", padding=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_kv_cache(out_path: Path) -> Path:
    """KV-cache precision tradeoff as a dual-axis grouped bar chart: prefill speed
    on the left axis (it spans ~33-885 t/s), decode on the right axis (a flat
    ~16-20 t/s). Symmetric formats keep prefill high; mixed K/V precision forces
    mid-execution conversion and collapses prefill while decode barely moves."""
    _set_advanced_theme()
    df = pd.read_csv(ADVANCED_DIR / "kv_cache_tradeoff.csv")
    df["combo"] = df["type_k"] + "/" + df["type_v"]

    prefill = df[df["n_prompt"] > 0][["combo", "avg_ts"]].rename(columns={"avg_ts": "prefill"})
    decode = df[df["n_gen"] > 0][["combo", "avg_ts"]].rename(columns={"avg_ts": "decode"})
    data = pd.merge(prefill, decode, on="combo").sort_values("prefill", ascending=False)

    combos = data["combo"].tolist()
    prefill_vals = data["prefill"].tolist()
    decode_vals = data["decode"].tolist()

    blue, red = "#4285F4", "#EA4335"
    x = range(len(combos))
    width = 0.42

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()

    # Draw blue (prefill) above red (decode); bars sit at offset x positions so
    # they never overlap, but ax1 needs a transparent patch + higher z-order.
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    bars_pre = ax1.bar([i - width / 2 for i in x], prefill_vals, width, color=blue,
                       label="Prefill Phase Speed (Tokens/s)", zorder=3)
    bars_dec = ax2.bar([i + width / 2 for i in x], decode_vals, width, color=red,
                       label="Decode Phase Speed (Tokens/s)", zorder=2)

    ax1.set_ylim(0, 1000)
    ax1.set_yticks([0, 250, 500, 750, 1000])
    ax2.set_ylim(0, 25)
    ax2.set_yticks([0, 5, 10, 15, 20, 25])

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(combos, rotation=45, ha="right")
    ax1.set_ylabel("Prefill Phase Speed (Tokens/s)")
    ax2.set_ylabel("Decode Phase Speed (Tokens/s)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)
    ax2.grid(False)

    ax1.bar_label(bars_pre, fmt="%.1f", padding=3, fontsize=9, color=blue)
    ax2.bar_label(bars_dec, fmt="%.1f", padding=3, fontsize=9, color=red)

    fig.legend(handles=[bars_pre, bars_dec], loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_offload_cliff(out_path: Path) -> Path:
    """Prefill and decode speed vs number of GPU layers offloaded (Llama 8B)."""
    _set_advanced_theme()
    df = pd.read_csv(ADVANCED_DIR / "ngl_test_8b.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    df_prompt = df[df["n_prompt"] > 0].sort_values("n_gpu_layers")
    df_gen = df[df["n_gen"] > 0].sort_values("n_gpu_layers")

    ax1.plot(df_prompt["n_gpu_layers"], df_prompt["avg_ts"], marker="o", color="teal", linewidth=2)
    ax1.set_title("Prefill Speed vs GPU Layers Offloaded (Llama 8B)")
    ax1.set_xlabel("Number of GPU Layers (-ngl)")
    ax1.set_ylabel("Tokens per Second")
    for x, y in zip(df_prompt["n_gpu_layers"], df_prompt["avg_ts"]):
        ax1.text(x, y + 25, f"{y:.1f}", ha="center", fontsize=9)

    ax2.plot(df_gen["n_gpu_layers"], df_gen["avg_ts"], marker="s", color="darkred", linewidth=2)
    ax2.set_title("Decode Speed vs GPU Layers Offloaded (Llama 8B)")
    ax2.set_xlabel("Number of GPU Layers (-ngl)")
    ax2.set_ylabel("Tokens per Second")
    for x, y in zip(df_gen["n_gpu_layers"], df_gen["avg_ts"]):
        ax2.text(x, y + 0.6, f"{y:.1f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_thread_sweep(out_path: Path) -> Path:
    """Prefill and decode speed vs CPU thread count (Mistral 24B)."""
    _set_advanced_theme()
    df = pd.read_csv(ADVANCED_DIR / "thread_optimization.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    df_prompt = df[df["n_prompt"] > 0].sort_values("n_threads")
    df_gen = df[df["n_gen"] > 0].sort_values("n_threads")

    ax1.plot(df_prompt["n_threads"], df_prompt["avg_ts"], marker="o", color="purple", linewidth=2)
    ax1.set_title("Prefill Speed vs Thread Count (Mistral 24B)")
    ax1.set_xlabel("Number of Compute Threads (-t)")
    ax1.set_ylabel("Tokens per Second")
    ax1.set_xticks(df_prompt["n_threads"])
    for x, y in zip(df_prompt["n_threads"], df_prompt["avg_ts"]):
        ax1.text(x, y + 2, f"{y:.1f}", ha="center", fontsize=9)

    ax2.plot(df_gen["n_threads"], df_gen["avg_ts"], marker="s", color="crimson", linewidth=2)
    ax2.set_title("Decode Speed vs Thread Count (Mistral 24B)")
    ax2.set_xlabel("Number of Compute Threads (-t)")
    ax2.set_ylabel("Tokens per Second")
    ax2.set_xticks(df_gen["n_threads"])
    for x, y in zip(df_gen["n_threads"], df_gen["avg_ts"]):
        ax2.text(x, y + 0.1, f"{y:.1f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# Registry maps a short key -> (function, default output filename in assets/).
CHARTS = {
    "engine_comparison": (plot_engine_comparison_demo, "engine_comparison.png"),
    "engine_comparison_full": (plot_engine_comparison, "engine_comparison_full.png"),
    "thread_sweep": (plot_thread_sweep, "thread_sweep.png"),
    "kv_cache": (plot_kv_cache, "kv_cache_quant.png"),
    "offload_cliff": (plot_offload_cliff, "offload_cliff.png"),
    "batch_size": (plot_batch_size, "batch_size.png"),
}