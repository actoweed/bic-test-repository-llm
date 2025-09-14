# run_tests.py
import os
import time
import csv
import json
import asyncio
import shutil
from pathlib import Path
import httpx

BASE = "http://127.0.0.1:8000"
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

GENERATE_TESTS = [
    {"prompt": "Напиши стих про Python", "model": "deepseek/deepseek-chat-v3.1:free", "max_tokens": 100},
    {"prompt": "Коротко опиши что такое API", "model": "z-ai/glm-4.5-air:free", "max_tokens": 80},
    {"prompt": "Реши 1+1 с объяснением", "model": "moonshotai/kimi-k2:free", "max_tokens": 50},
    {"prompt": "Скажи привет простыми словами", "model": "deepseek/deepseek-chat-v3.1:free", "max_tokens": 20},
    {"prompt": "Напиши короткое мотивационное сообщение", "model": "z-ai/glm-4.5-air:free", "max_tokens": 60},
]

BENCHMARK_TESTS = [
    {"prompts_file": "prompts.txt", "model": "deepseek/deepseek-chat-v3.1:free", "runs": 3},
    {"prompts_file": "prompts.txt", "model": "moonshotai/kimi-k2:free", "runs": 5},
    {"prompts_file": "prompts.txt", "model": "z-ai/glm-4.5-air:free", "runs": 2},
    {"prompts_file": "prompts.txt", "model": "deepseek/deepseek-chat-v3.1:free", "runs": 4},
    {"prompts_file": "prompts.txt", "model": "moonshotai/kimi-k2:free", "runs": 3},
]

async def run_generate(client, idx, test):
    url = f"{BASE}/generate"
    payload = {"prompt": test["prompt"], "model": test["model"], "max_tokens": test.get("max_tokens", 512)}
    start = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=120)
        elapsed = time.perf_counter() - start
        text = await resp.text()
        try:
            j = resp.json()
        except Exception:
            j = {"raw": text}
        return {"type": "generate", "idx": idx, "model": test["model"], "prompt": test["prompt"], "status": resp.status_code, "elapsed": elapsed, "response": j}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"type": "generate", "idx": idx, "model": test["model"], "prompt": test["prompt"], "status": "error", "elapsed": elapsed, "error": str(e)}

async def run_benchmark(client, idx, test):
    url = f"{BASE}/benchmark"
    files = {"prompt_file": open(test["prompts_file"], "rb")}
    data = {"model": test["model"], "runs": str(test["runs"])}
    start = time.perf_counter()
    try:
        resp = await client.post(url, data=data, files=files, timeout=600)
        elapsed = time.perf_counter() - start
        text = await resp.text()
        try:
            j = resp.json()
        except Exception:
            j = {"raw": text}
        # If server saved benchmark_results.csv, try download / copy local (server writes to disk server-side)
        return {"type": "benchmark", "idx": idx, "model": test["model"], "runs": test["runs"], "status": resp.status_code, "elapsed": elapsed, "response": j}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"type": "benchmark", "idx": idx, "model": test["model"], "runs": test["runs"], "status": "error", "elapsed": elapsed, "error": str(e)}
    finally:
        try:
            files["prompt_file"].close()
        except Exception:
            pass

async def main():
    async with httpx.AsyncClient() as client:
        results = []
        # Run generate tests sequentially (or concurrently if desired)
        for i, t in enumerate(GENERATE_TESTS, start=1):
            print(f"Running generate test {i}/{len(GENERATE_TESTS)} -> model={t['model']}")
            r = await run_generate(client, i, t)
            results.append(r)

        # Run benchmark tests (sequential)
        for i, t in enumerate(BENCHMARK_TESTS, start=1):
            print(f"Running benchmark test {i}/{len(BENCHMARK_TESTS)} -> model={t['model']} runs={t['runs']}")
            r = await run_benchmark(client, i, t)
            results.append(r)

    # Save results summary
    csv_path = ARTIFACTS_DIR / "test_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        fieldnames = ["type","idx","model","prompt","runs","status","elapsed","response","error"]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "type": r.get("type"),
                "idx": r.get("idx"),
                "model": r.get("model"),
                "prompt": r.get("prompt", ""),
                "runs": r.get("runs", ""),
                "status": r.get("status"),
                "elapsed": r.get("elapsed"),
                "response": json_safe(r.get("response")),
                "error": r.get("error", "")
            })

    # Try to copy server artifacts if present
    for name in ("benchmark_results.csv", "server_logs.txt"):
        if os.path.exists(name):
            shutil.copy2(name, ARTIFACTS_DIR / name)
            print(f"Copied {name} -> {ARTIFACTS_DIR / name}")
        else:
            print(f"Server-side artifact {name} not found in current dir (server may run in different container).")

    print(f"Test run complete. Artifacts in {ARTIFACTS_DIR.resolve()} and summary CSV {csv_path}")

def json_safe(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

if __name__ == "__main__":
    asyncio.run(main())
