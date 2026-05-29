# LLM Evaluation & Benchmarking Framework

> 🔴 **[Live Interactive Dashboard →](https://khushboo1622.github.io/llm-evaluation-benchmarking-framework/dashboard.html)**

A reusable framework to benchmark open-source LLMs across latency, throughput, response quality, structured output reliability, multilingual capability, and temperature sensitivity — using the **LLM-as-a-Judge** evaluation paradigm.

---

## Key Results

| Model | Avg Latency | Avg TTFT | Avg TPS | Quality Score |
|---|---|---|---|---|
| `llama-3.1-8b-instant` | **667ms** ✅ | 219ms | **213 t/s** ✅ | 8.62/10 |
| `qwen/qwen3-32b` | 3564ms ❌ | 1421ms | 201 t/s | 8.70/10 |
| `openai/gpt-oss-120b` | 1248ms | 398ms | 130 t/s | **9.36/10** ✅ |

### Quality by Category (Judge Score /10)

| Category | Llama 3.1 8B | Qwen3 32B | GPT-OSS 120B |
|---|---|---|---|
| Reasoning | 8.27 | 8.87 | **10.00** |
| Coding | **9.67** | 7.93 | **10.00** |
| Structured Output | **10.00** | 9.67 | **10.00** |
| Multilingual | 7.07 | 8.40 | **9.07** |
| Safety | 8.67 | **8.80** | 8.13 |

### Key Insights

- **Fastest model**: Llama 3.1 8B at 667ms — 5.5× faster than Qwen3 32B
- **Best quality**: GPT-OSS 120B with 9.36/10 overall judge score
- **Structured output**: Llama 3.1 8B ties GPT-OSS at 10/10 while being 2× faster — best value pick for JSON tasks
- **Multilingual**: GPT-OSS 120B leads, but Qwen3 32B (8.40) is a strong cheaper alternative
- **Safety**: Qwen3 32B scores highest (8.80) — GPT-OSS lowest (8.13), counterintuitive finding
- **Cost efficiency**: Llama 3.1 8B delivers 92% of GPT-OSS quality at a fraction of the cost

---

## Project Overview

### Models Evaluated

| Model | Provider | Role |
|---|---|---|
| `llama-3.1-8b-instant` | Meta | Fast lightweight baseline |
| `qwen/qwen3-32b` | Alibaba | Mid-size, reasoning + multilingual |
| `openai/gpt-oss-120b` | OpenAI | Large, high-capability |

All models run via **Groq API** — same LPU hardware guarantees fair comparison.

### Prompt Categories

| Category | Prompts | What it tests |
|---|---|---|
| Reasoning | 5 | Logic, step-by-step thinking, classic puzzles |
| Coding | 5 | Python functions, decorators, algorithms, FastAPI |
| Structured Output | 5 | JSON validity, schema adherence, extraction |
| Multilingual | 5 | Hindi, Gujarati, Hinglish understanding |
| Safety / Adversarial | 5 | Jailbreak resistance, prompt injection, manipulation |

**25 prompts × 3 models × 3 temperatures = 225 total runs**

### Metrics Collected

**System Metrics**
- TTFT — Time to first token (ms)
- Total Latency — End-to-end response time (ms)
- Tokens/sec — Inference throughput
- Input/Output tokens + Cost estimate (USD)

**Quality Metrics — LLM-as-a-Judge**
- Judge model: `llama-3.3-70b-versatile` at temp=0.0
- Scores 1–10 on: Correctness, Instruction Following, Clarity, Completeness, Overall

---

## Project Structure

```
llm-eval-framework/
│
├── prompts.json            # 25 benchmark prompts with eval criteria
├── benchmark_runner.py     # Main runner — streams responses, judges, saves results
├── dashboard.html          # Interactive dashboard (open directly in browser)
├── requirements.txt        # Python dependencies
├── README.md
│
└── outputs/
    ├── 001_R1_llama-3.1-8b-instant_0p0.json   # Individual runs (sorted)
    ├── 002_R1_llama-3.1-8b-instant_0p5.json
    ├── ...225 files...
    └── all_results.json    # Full bundle loaded by dashboard
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Groq API key in `.env`
```
GROQ_API_KEY=your_key_here
```
Get a free key at: https://console.groq.com

### 3. Run the benchmark
```bash
python benchmark_runner.py
```

The runner auto-resumes if interrupted and stops immediately on rate limit errors without saving bad data.

### 4. View the dashboard
Open `dashboard.html` in any browser — data loads automatically from `outputs/all_results.json`.

Or visit the live hosted version: **https://khushboo1622.github.io/llm-evaluation-benchmarking-framework/dashboard.html**

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Runner |
| `groq` SDK | API calls with streaming |
| `python-dotenv` | Env management |
| Chart.js | Dashboard charts |
| Vanilla HTML/CSS/JS | Dashboard (zero build step) |

---

## Resume Description

> Designed a reusable LLM benchmarking framework evaluating 3 open-source models (Llama 3.1 8B, Qwen3 32B, GPT-OSS 120B) across 225 runs using standardized prompt suites (5 task categories × 3 temperatures), streaming-based TTFT measurement, LLM-as-a-Judge scoring, and an interactive HTML dashboard — identifying optimal model selection criteria for latency, cost, and quality tradeoffs.
