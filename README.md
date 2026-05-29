# LLM Evaluation & Benchmarking Framework

A reusable framework to benchmark open-source LLMs across latency, throughput, response quality, structured output reliability, multilingual capability, and temperature sensitivity.

---

## Models Evaluated

| Model | Provider | Role |
|---|---|---|
| `llama-3.1-8b-instant` | Meta | Fast lightweight baseline |
| `qwen/qwen3-32b` | Alibaba | Mid-size, reasoning + multilingual |
| `openai/gpt-oss-120b` | OpenAI | Large, high-capability |

All models run via **Groq API** (same LPU hardware = fair comparison).

---

## Prompt Categories

| Category | Prompts | What it tests |
|---|---|---|
| Reasoning | 5 | Logic, step-by-step thinking |
| Coding | 5 | Python functions, APIs, algorithms |
| Structured Output | 5 | JSON validity, schema adherence |
| Multilingual | 5 | Hindi, Gujarati, Hinglish |
| Safety / Adversarial | 5 | Jailbreak resistance, alignment |

**Total: 25 prompts × 3 models × 3 temperatures = 225 runs**

---

## Metrics Collected

### System Metrics
- **TTFT** — Time to first token (ms)
- **Total Latency** — End-to-end response time (ms)
- **Tokens/sec** — Inference throughput
- **Input/Output tokens** — Token counts
- **Cost estimate** — USD per API call

### Quality Metrics (LLM-as-a-Judge)
Judge model: `llama-3.3-70b-versatile` at temperature 0.0

Scores each response 1–10 on:
- Correctness
- Instruction Following
- Clarity
- Completeness
- Overall

---

## Project Structure

```
llm-eval-framework/
│
├── prompts.json          # 25 benchmark prompts (5 categories × 5 prompts)
├── benchmark_runner.py   # Main async runner — generates all_results.json
├── dashboard.html        # Interactive HTML dashboard (open in browser)
├── requirements.txt      # Python dependencies
├── README.md
│
└── outputs/              # Auto-created by runner
    ├── R1_llama_0p0.json # Individual run results
    ├── R1_llama_0p5.json
    ├── ...
    └── all_results.json  # Full results bundle → load into dashboard
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key
```bash
export GROQ_API_KEY=your_key_here
```
Get a free key at: https://console.groq.com

### 3. Run the benchmark
```bash
python benchmark_runner.py
```

This will:
- Run all 225 evaluations (resumes if interrupted)
- Print progress as it goes
- Save each result individually to `outputs/`
- Save full bundle to `outputs/all_results.json`

Estimated time: ~15–30 minutes depending on rate limits.

### 4. View the dashboard
1. Open `dashboard.html` in any browser
2. Click **Upload Results JSON**
3. Select `outputs/all_results.json`
4. Explore charts, filters, and per-run responses

Or click **Load Sample Data** to preview the dashboard instantly.

---

## Dashboard Features

- **Filter** by model, category, temperature
- **Latency & TPS** bar charts per model
- **Quality vs Speed** scatter plot (key insight chart)
- **Temperature vs Score** line chart
- **TTFT** comparison
- **Cost estimation** per model
- **Heatmap** — model × category quality scores
- **Raw results table** with search + sort
- **Response viewer** modal per run

---

## Key Insights (to document in your report)

Answer these from your results:

1. **Which model is fastest?** → Latency + TPS charts
2. **Which model has best quality?** → Judge score heatmap
3. **Best quality/speed tradeoff?** → Scatter plot
4. **Does temperature affect quality?** → Temp line chart
5. **Which model handles multilingual best?** → Heatmap row
6. **Which model is most cost-efficient?** → Cost chart
7. **Which model resists jailbreaks best?** → Safety category scores

---

## Resume Support

If the runner is interrupted, re-run `benchmark_runner.py`. It checks for existing output files and **skips completed runs** automatically.

---

## Judge Model Note

The judge (`llama-3.3-70b-versatile`) is separate from the 3 evaluated models. It uses a fixed rubric and temperature=0.0 for deterministic, consistent scoring. This follows the **LLM-as-a-Judge** paradigm used in research.

---

## Tech Stack

- Python 3.10+
- `httpx` — async HTTP client
- Groq API — inference backend (free tier)
- Vanilla HTML/CSS/JS + Chart.js — dashboard (no build step)
