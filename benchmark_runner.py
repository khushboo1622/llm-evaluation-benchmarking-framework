"""
LLM Evaluation & Benchmarking Framework
Runs 25 prompts across 3 models x 3 temperatures = 225 evaluations
Uses official groq SDK with streaming for accurate TTFT measurement
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODELS = [
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "openai/gpt-oss-120b",
]

JUDGE_MODEL = "llama-3.3-70b-versatile"

TEMPERATURES = [0.0, 0.5, 1.0]

COST_MAP = {
    "llama-3.1-8b-instant":    {"input": 0.05,  "output": 0.08},
    "qwen/qwen3-32b":           {"input": 0.29,  "output": 0.39},
    "openai/gpt-oss-120b":      {"input": 3.00,  "output": 6.00},
    "llama-3.3-70b-versatile":  {"input": 0.59,  "output": 0.79},
}

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPTS_FILE = Path("prompts.json")

# Errors that mean we hit a limit — stop immediately
FATAL_ERROR_KEYWORDS = [
    "rate_limit_exceeded",
    "tokens per minute",
    "requests per minute",
    "requests per day",
    "exceeded",
    "429",
    "quota",
    "limit",
]


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def load_prompts():
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def estimate_cost(model, input_tokens, output_tokens):
    rates = COST_MAP.get(model, {"input": 0, "output": 0})
    return round(
        (input_tokens / 1_000_000) * rates["input"] +
        (output_tokens / 1_000_000) * rates["output"],
        8
    )


def is_rate_limit_error(error_str: str) -> bool:
    lower = error_str.lower()
    return any(kw in lower for kw in FATAL_ERROR_KEYWORDS)


def save_result(result: dict, run_number: int):
    # Filename format: 001_R1_llama-3.1-8b-instant_0p0.json
    # Zero-padded run number ensures correct sort order in folder
    safe_model = result["model"].replace("/", "_")
    safe_temp = str(result["temperature"]).replace(".", "p")
    filename = f"{run_number:03d}_{result['prompt_id']}_{safe_model}_{safe_temp}.json"
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return filename


def get_run_filename(run_number: int, prompt_id: str, model: str, temp: float) -> str:
    safe_model = model.replace("/", "_")
    safe_temp = str(temp).replace(".", "p")
    return f"{run_number:03d}_{prompt_id}_{safe_model}_{safe_temp}.json"


# ─────────────────────────────────────────
# API CALL
# ─────────────────────────────────────────

def call_groq(model: str, prompt: str, temperature: float) -> dict:
    t_start = time.perf_counter()
    ttft = None
    chunks = []

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                if ttft is None:
                    ttft = round((time.perf_counter() - t_start) * 1000, 2)
                chunks.append(delta)

        total_latency = round((time.perf_counter() - t_start) * 1000, 2)
        response_text = "".join(chunks)
        output_tokens = len(response_text.split())
        input_tokens = len(prompt.split())

        return {
            "success": True,
            "response": response_text,
            "ttft_ms": ttft or total_latency,
            "latency_ms": total_latency,
            "tokens_per_second": round(output_tokens / (total_latency / 1000), 2) if total_latency > 0 else 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response_length": len(response_text),
            "cost_usd": estimate_cost(model, input_tokens, output_tokens),
            "error": None,
            "rate_limited": False,
        }

    except Exception as e:
        error_str = str(e)
        return {
            "success": False,
            "response": "",
            "ttft_ms": None,
            "latency_ms": None,
            "tokens_per_second": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "response_length": 0,
            "cost_usd": 0,
            "error": error_str,
            "rate_limited": is_rate_limit_error(error_str),
        }


# ─────────────────────────────────────────
# LLM JUDGE
# ─────────────────────────────────────────

JUDGE_SYSTEM = """You are an impartial LLM evaluator. Given a prompt, expected criteria, and a model response,
score the response on these dimensions from 1-10. Return ONLY valid JSON, no explanation, no markdown.

{"correctness": 0, "instruction_following": 0, "clarity": 0, "completeness": 0, "overall": 0}"""


def judge_response(prompt: str, criteria: str, response: str) -> dict:
    empty = {"correctness": 0, "instruction_following": 0, "clarity": 0, "completeness": 0, "overall": 0}

    if not response.strip():
        return empty

    judge_prompt = f"""Prompt given to model:
{prompt}

Expected criteria:
{criteria}

Model response:
{response}

Score this response. Return ONLY JSON."""

    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        error_str = str(e)
        if is_rate_limit_error(error_str):
            raise RuntimeError(f"RATE_LIMIT_IN_JUDGE: {error_str}")
        print(f"  ⚠ Judge error: {e}")
        return empty


# ─────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────

def build_bundle():
    """Rebuild all_results.json from individual files in outputs/"""
    files = sorted(OUTPUT_DIR.glob("*.json"))
    results = []
    for f in files:
        if f.name == "all_results.json":
            continue
        try:
            with open(f, encoding="utf-8") as fp:
                results.append(json.load(fp))
        except Exception:
            pass
    bundle_path = OUTPUT_DIR / "all_results.json"
    with open(bundle_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    return len(results)


def run_benchmark():
    prompts = load_prompts()
    all_results = []

    total = len(prompts) * len(MODELS) * len(TEMPERATURES)
    done = 0

    print(f"\n{'='*60}")
    print(f"  LLM Benchmark Runner")
    print(f"  {len(prompts)} prompts x {len(MODELS)} models x {len(TEMPERATURES)} temps = {total} runs")
    print(f"{'='*60}\n")

    for prompt_item in prompts:
        for model in MODELS:
            for temp in TEMPERATURES:
                done += 1
                run_number = done
                filename = get_run_filename(run_number, prompt_item["id"], model, temp)
                result_path = OUTPUT_DIR / filename

                print(f"[{done:03d}/{total}] {prompt_item['id']} | {model} | temp={temp}")

                # Resume support
                if result_path.exists():
                    try:
                        with open(result_path, encoding="utf-8") as f:
                            all_results.append(json.load(f))
                        print(f"  ↳ Skipping (already exists)")
                        continue
                    except (json.JSONDecodeError, ValueError):
                        print(f"  ↳ Corrupted file, re-running...")
                        result_path.unlink()

                # Run model
                metrics = call_groq(model, prompt_item["prompt"], temp)

                # Check for rate limit — stop immediately, don't save
                if metrics["rate_limited"]:
                    print(f"\n{'='*60}")
                    print(f"  ⛔ RATE LIMIT HIT at run {done}/{total}")
                    print(f"  Error: {metrics['error']}")
                    print(f"  ↳ Stopping safely. {done-1} runs saved.")
                    print(f"  ↳ Re-run the script later to resume from run {done}.")
                    print(f"{'='*60}\n")
                    n = build_bundle()
                    print(f"  Bundle saved: {n} results → outputs/all_results.json")
                    return

                print(f"  ↳ Got response ({metrics['response_length']} chars) | {metrics['latency_ms']}ms")

                # Judge — also stop if rate limited
                print(f"  ↳ Judging...")
                try:
                    scores = judge_response(
                        prompt_item["prompt"],
                        prompt_item["eval_criteria"],
                        metrics["response"],
                    )
                except RuntimeError as e:
                    if "RATE_LIMIT_IN_JUDGE" in str(e):
                        print(f"\n{'='*60}")
                        print(f"  ⛔ RATE LIMIT HIT (judge) at run {done}/{total}")
                        print(f"  ↳ Stopping safely. {done-1} runs saved.")
                        print(f"  ↳ Re-run the script later to resume from run {done}.")
                        print(f"{'='*60}\n")
                        n = build_bundle()
                        print(f"  Bundle saved: {n} results → outputs/all_results.json")
                        return
                    scores = {"correctness": 0, "instruction_following": 0, "clarity": 0, "completeness": 0, "overall": 0}

                print(f"  ↳ Score: {scores.get('overall', '?')}/10")

                result = {
                    "run_id": filename.replace(".json", ""),
                    "run_number": run_number,
                    "prompt_id": prompt_item["id"],
                    "category": prompt_item["category"],
                    "difficulty": prompt_item["difficulty"],
                    "model": model,
                    "temperature": temp,
                    "prompt": prompt_item["prompt"],
                    "eval_criteria": prompt_item["eval_criteria"],
                    "timestamp": datetime.utcnow().isoformat(),
                    **metrics,
                    "judge_scores": scores,
                }

                save_result(result, run_number)
                all_results.append(result)

                time.sleep(0.5)

    # All done — save final bundle
    n = build_bundle()
    print(f"\n✅ All {n} results saved to outputs/all_results.json")


if __name__ == "__main__":
    run_benchmark()