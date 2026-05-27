# 🖼️ Развёртывание модели классификации изображений на NVIDIA Triton Inference Server

## 📝 Описание

Данный репозиторий содержит проект для развёртывания модели классификации изображений с использованием **NVIDIA Triton Inference Server**.

В качестве серверной части используется **FastAPI Gateway**, который принимает изображение от пользователя, выполняет предварительную обработку (ресайз до 64×64, нормализация /255.0) и отправляет данные в Triton Inference Server. Для мониторинга работы системы подключены **Prometheus** и **Grafana**.

Модель классифицирует изображения на три класса:

- Курица (`chicken`)
- Корова (`cow`)
- Лошадь (`horse`)

---

## 🚀 Используемые технологии

- **Python** — основной язык проекта
- **TensorFlow / Keras** — исходная модель классификации
- **NVIDIA Triton Inference Server** — сервер инференса модели
- **FastAPI** — API Gateway для обработки HTTP-запросов
- **Prometheus** — сбор метрик
- **Grafana** — визуализация метрик

---

## 📁 Структура проекта

```text
triton-classifier/
├── docker-compose.yml
├── load_test.py
├── README.md
│
├── model_repository/
│   └── image_classifier/
│       ├── config.pbtxt
│       └── 1/
│           └── model.onnx
│
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   └── classes.pkl
│
├── prometheus/
│   └── prometheus.yml
│
└── grafana/
    └── provisioning/
        ├── datasources/
        │   └── datasources.yml
        └── dashboards/
            ├── dashboards.yml
            └── triton.json
```

---

## 🛠️ Установка и запуск

Клонируйте репозиторий:

```bash
git clone https://github.com/vadim13213/triton-animal-classifier.git
cd triton-animal-classifier
```

Запустите все сервисы:

```
docker compose up -d --build
```

Проверьте статус контейнеров:

```
docker compose ps
```

После запуска должны быть доступны сервисы:

| Сервис | URL |
|---|---|
| FastAPI Swagger UI | http://localhost:8080/docs |
| Triton HTTP API | http://localhost:8000 |
| Triton metrics | http://localhost:8002/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Логин и пароль для Grafana:

```
admin / admin
```

---

## 🔌 API

### Проверка состояния

```bash
curl http://localhost:8080/health
```

### Классификация изображения

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@example.jpg"
```

Пример ответа:

```json
{
  "predicted_class": "horse",
  "confidence": 0.95,
  "probabilities": {
    "chicken": 0.01,
    "cow": 0.04,
    "horse": 0.95
  }
}
```

---

## 📊 Мониторинг

Prometheus собирает метрики Triton с порта:

```text
http://localhost:8002/metrics
```

Grafana доступна по адресу:

```text
http://localhost:3000
```

---

## 🧪 Нагрузочное тестирование

Нагрузочное тестирование выполняется скриптом:

```bash
python load_test.py
```

Параметры тестирования:

| Параметр | Значение |
|---|---|
| Количество запросов | 1000 |
| Параллельных запросов | 50 |
| Формат входа | JPEG-изображение 64×64 |
| API endpoint | `http://localhost:8080/predict` |
| Режим инференса | CPU |

### Результаты тестирования Dynamic Batching

| Конфигурация | Avg Latency (ms) | P95 (ms) | P99 (ms) | Throughput (RPS) |
|--------------|------------------|----------|----------|------------------|
| Без батчинга | 472.78 | 584.14 | 4500.50 | 103.69 |
| Малый батч | 2835.23 | 3592.13 | 19029.86 | 16.31 |
| Средний батч | 4093.89 | 5024.94 | 22401.68 | 8.64 |
| Большой батч | 5617.33 | 6921.80 | 28125.59 | 4.65 |

### Выводы

Динамический батчинг на CPU **значительно ухудшает производительность**: средняя задержка возрастает с 473 мс до 5,6 секунд, а пропускная способность падает со 104 до 4,7 RPS. Это связано с тем, что накопление батча добавляет задержки без выигрыша от параллелизации (эффективно только на GPU). **Рекомендуется отключать динамический батчинг для CPU-инференса.**
```

Теперь вы можете:

1. Заменить содержимое вашего локального `README.md` на этот текст.
2. Сохранить файл.
3. Выполнить в командной строке:

```bash
git add README.md
git commit -m "Update README with project results"
git push origin master
```

После этого репозиторий будет содержать актуальный и хорошо оформленный README, готовый для сдачи. Если нужны ещё правки – сообщите.
