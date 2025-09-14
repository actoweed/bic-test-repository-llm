# main.py — Level 2 (асинхронный, качественный)
import os
import time
import csv
import math
import statistics
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Tuple
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
import httpx
import asyncio

# Load env
load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    # allow startup but clearly fail on calls
    pass

# Logging: rotate errors to server_logs.txt
logger = logging.getLogger('server')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('server_logs.txt', maxBytes=5_000_000, backupCount=3, encoding='utf-8')
handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI(title='FastAPI LLM Benchmark — Level 2')

MODELS = [
    "deepseek/deepseek-chat-v3.1:free",
    "z-ai/glm-4.5-air:free",
    "moonshotai/kimi-k2:free",
]

OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'

# Shared httpx client
client: Optional[httpx.AsyncClient] = None

@app.on_event('startup')
async def startup_event():
    global client
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=50)
    client = httpx.AsyncClient(timeout=60.0, limits=limits)

@app.on_event('shutdown')
async def shutdown_event():
    global client
    if client:
        await client.aclose()

def _extract_text_and_tokens(data: dict) -> Tuple[str, Optional[int]]:
    """Try to extract text/token usage from various possible upstream shapes."""
    text = ''
    tokens = None
    try:
        if 'choices' in data and data['choices']:
            ch = data['choices'][0]
            if isinstance(ch.get('message'), dict):
                text = ch['message'].get('content', '')
            else:
                text = ch.get('text') or ch.get('message') or ''
        else:
            text = data.get('output') or data.get('result') or ''
    except Exception:
        text = str(data)
    try:
        tokens = data.get('usage', {}).get('total_tokens')
    except Exception:
        tokens = None
    return text, tokens

async def call_openrouter_async(payload: dict, max_retries: int = 5, initial_backoff: float = 0.5):
    """Call OpenRouter (OpenAI-compatible) with exponential backoff + jitter and 429 handling."""
    global client
    if client is None:
        raise RuntimeError('HTTP client not initialized')

    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json',
    }

    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.post(OPENROUTER_API_URL, json=payload)
        except httpx.RequestError as e:
            logger.error(f'RequestError attempt {attempt}: {e}')
            if attempt == max_retries:
                raise HTTPException(status_code=502, detail='Upstream request failed')
            # jitter: small random factor using os.urandom
            jitter = (os.urandom(1)[0] / 255)
            await asyncio.sleep(backoff * (1 + jitter))
            backoff *= 2
            continue

        if resp.status_code == 429:
            retry_after = resp.headers.get('Retry-After')
            try:
                wait = float(retry_after) if retry_after else backoff
            except Exception:
                wait = backoff
            logger.error(f'429 from upstream, attempt {attempt}, waiting {wait}s')
            if attempt == max_retries:
                detail = None
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                raise HTTPException(status_code=429, detail={'message': 'Rate limited', 'detail': detail})
            jitter = (os.urandom(1)[0] / 255)
            await asyncio.sleep(wait * (1 + jitter))
            backoff *= 2
            continue

        if 500 <= resp.status_code < 600:
            logger.error(f'Upstream 5xx: {resp.status_code} body={resp.text}')
            if attempt == max_retries:
                raise HTTPException(status_code=502, detail='Upstream server error')
            await asyncio.sleep(backoff)
            backoff *= 2
            continue

        # success or client error (4xx)
        return resp

    raise HTTPException(status_code=500, detail='Retries exhausted')

@app.get('/models')
async def get_models():
    return {'models': MODELS}

@app.post('/generate')
async def generate(request: Request):
    """
    Body JSON: {"prompt": "...", "model": "...", "max_tokens": 512}
    Returns JSON: {response, tokens_used, latency_seconds}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid JSON body')

    prompt = body.get('prompt')
    model = body.get('model')
    try:
        max_tokens = int(body.get('max_tokens', 512))
    except Exception:
        max_tokens = 512

    if not prompt or not model:
        raise HTTPException(status_code=400, detail='Missing prompt or model')

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
    }

    start = time.perf_counter()
    try:
        resp = await call_openrouter_async(payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Unhandled error in /generate: {e}')
        raise HTTPException(status_code=500, detail='Internal error')

    elapsed = time.perf_counter() - start

    if resp.status_code >= 400:
        logger.error(f'Upstream returned error {resp.status_code}: {resp.text}')
        raise HTTPException(status_code=502, detail='Upstream error')

    try:
        data = resp.json()
    except Exception:
        data = {'raw': resp.text}

    text, tokens = _extract_text_and_tokens(data)

    return JSONResponse({'response': text, 'tokens_used': tokens, 'latency_seconds': elapsed})

@app.post('/benchmark')
async def benchmark(prompt_file: UploadFile = File(...), model: str = Form(...), runs: int = Form(5)):
    """
    Multipart/form-data: prompt_file (txt), model, runs
    Returns JSON: summary stats; saves benchmark_results.csv
    """
    if prompt_file.content_type != 'text/plain' and not prompt_file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail='prompt_file must be a text file (.txt)')

    raw = await prompt_file.read()
    try:
        text = raw.decode('utf-8')
    except Exception:
        text = raw.decode('latin-1')

    prompts = [line.strip() for line in text.splitlines() if line.strip()]
    if not prompts:
        raise HTTPException(status_code=400, detail='No prompts found')

    results = []

    # concurrency control: limit concurrent requests
    semaphore = asyncio.Semaphore(8)

    async def run_single(prompt: str, run_idx: int):
        payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 512}
        async with semaphore:
            start = time.perf_counter()
            try:
                resp = await call_openrouter_async(payload)
            except HTTPException as e:
                logger.error(f'Benchmark call failed prompt="{prompt}" run={run_idx} detail={e.detail}')
                return {'model': model, 'prompt': prompt, 'run': run_idx, 'latency_seconds': math.nan, 'tokens_used': None}
            elapsed = time.perf_counter() - start
            tokens = None
            try:
                data = resp.json()
                _, tokens = _extract_text_and_tokens(data)
            except Exception:
                tokens = None
            return {'model': model, 'prompt': prompt, 'run': run_idx, 'latency_seconds': elapsed, 'tokens_used': tokens}

    # build tasks
    tasks = []
    for p in prompts:
        for r in range(1, int(runs) + 1):
            tasks.append(run_single(p, r))

    # run tasks concurrently
    gathered = await asyncio.gather(*tasks)
    results.extend(gathered)

    # save CSV
    csv_fname = 'benchmark_results.csv'
    with open(csv_fname, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=['model', 'prompt', 'run', 'latency_seconds', 'tokens_used'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # aggregate stats
    grouped = {}
    for row in results:
        key = (row['model'], row['prompt'])
        grouped.setdefault(key, []).append(row['latency_seconds'])

    summary = []
    for (m, p), lat_list in grouped.items():
        clean = [x for x in lat_list if not (isinstance(x, float) and math.isnan(x))]
        if clean:
            avg = statistics.mean(clean)
            mn = min(clean)
            mx = max(clean)
            sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
        else:
            avg = mn = mx = sd = math.nan
        summary.append({'model': m, 'prompt': p, 'avg': avg, 'min': mn, 'max': mx, 'std_dev': sd})

    return JSONResponse({'summary': summary, 'csv': csv_fname})

@app.exception_handler(HTTPException)
async def http_exc_handler(request, exc):
    # Log 5xx and rate-limits
    try:
        status_code = exc.status_code
    except Exception:
        status_code = 500
    if status_code >= 500 or status_code == 429:
        logger.error(f'HTTPException {status_code}: {exc.detail}')
    return JSONResponse(status_code=status_code, content={'detail': exc.detail})
