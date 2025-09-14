# 🚀 FastAPI + LLM Benchmark (Level 2)

Тестовое задание: сервер для работы с LLM через [OpenRouter](https://openrouter.ai/), 
с поддержкой генерации, бенчмарков и логированием.

---

## 📂 Структура проекта

fastapi-llm-benchmark/
├── main.py # FastAPI сервер
├── requirements.txt # зависимости
├── .env.example # пример env-файла (API-ключ)
├── prompts.txt # пример промптов для бенчмарка
├── run_tests.py # автоматический прогон тестов (5 generate + 5 benchmark)
├── Dockerfile # (опционально) контейнеризация
├── docker-compose.yml # (опционально) запуск через docker-compose
└── README.md # этот файл

yaml


---

## ⚙️ Установка и запуск (локально)

1. **Склонировать проект и перейти в папку**
   ```bash
   git clone <repo-url> fastapi-llm-benchmark
   cd fastapi-llm-benchmark
Создать .env

bash

cp .env.example .env
и вставить свой ключ OpenRouter:

ini

OPENROUTER_API_KEY=ваш_ключ
Создать виртуальное окружение и установить зависимости

bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
Запустить сервер

bash

uvicorn main:app --host 127.0.0.1 --port 8000
🔥 Примеры запросов (cURL)
1. Список моделей
bash

curl -s "http://127.0.0.1:8000/models"
2. Генерация текста
bash

curl -sS -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Напиши стих про Python","model":"deepseek/deepseek-chat-v3.1:free","max_tokens":100}'
3. Benchmark (multipart/form-data)
bash

curl -sS -X POST "http://127.0.0.1:8000/benchmark" \
  -F "prompt_file=@prompts.txt;type=text/plain" \
  -F "model=deepseek/deepseek-chat-v3.1:free" \
  -F "runs=3"
📊 Автоматические тесты
Проект включает скрипт run_tests.py, который выполняет:

5 тестов на /generate (разные модели, разные промпты);

5 тестов на /benchmark (разные модели и количество прогонов).

Запуск
В отдельном терминале (сервер уже должен быть запущен):

bash

python run_tests.py
Результаты
все артефакты сохраняются в папку artifacts/;

там будут:

test_results.csv — сводка по 10 тестам;

копии benchmark_results.csv и server_logs.txt (если они есть на сервере).

🐳 Запуск в Docker (опционально)
Собрать образ

bash

docker build -t llm-benchmark .
Запуск через docker-compose

bash

docker-compose up --build
Сервер будет доступен на http://127.0.0.1:8000.

📂 Артефакты
server_logs.txt — ошибки с ротацией.

benchmark_results.csv — результаты последнего бенчмарка.

artifacts/ — результаты автотестов.

✅ Проверка latency
/generate
bash

curl -o /dev/null -sS -w "time_total=%{time_total}\n" \
  -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Скажи привет","model":"deepseek/deepseek-chat-v3.1:free"}'
/benchmark
bash

curl -o /dev/null -sS -w "time_total=%{time_total}\n" \
  -X POST "http://127.0.0.1:8000/benchmark" \
  -F "prompt_file=@prompts.txt;type=text/plain" \
  -F "model=deepseek/deepseek-chat-v3.1:free" \
  -F "runs=3"