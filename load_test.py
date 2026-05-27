"""
Скрипт нагрузочного тестирования API (Triton + FastAPI)
Отправляет изображения через multipart/form-data, измеряет latency и throughput.
"""

import asyncio
import aiohttp
import time
import numpy as np
from PIL import Image
import io

# === НАСТРОЙКИ ===
API_URL = "http://localhost:8080/predict"
TOTAL_REQUESTS = 1000      # количество запросов
CONCURRENT = 50            # параллельных запросов

def generate_test_image() -> bytes:
    """Генерирует случайное изображение 64x64 RGB и возвращает байты JPEG"""
    img = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()

async def send_request(session, semaphore, image_bytes):
    """Отправляет один запрос с файлом, возвращает (успех, время_в_секундах)"""
    async with semaphore:
        start = time.perf_counter()
        try:
            data = aiohttp.FormData()
            data.add_field('file', image_bytes, filename='test.jpg', content_type='image/jpeg')
            async with session.post(API_URL, data=data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                await resp.json()
                success = resp.status == 200
                elapsed = time.perf_counter() - start
                return success, elapsed
        except Exception:
            return False, time.perf_counter() - start

async def main():
    print(f"🚀 Запуск теста: {TOTAL_REQUESTS} запросов, {CONCURRENT} параллельных\n")
    
    image_bytes = generate_test_image()
    semaphore = asyncio.Semaphore(CONCURRENT)
    
    async with aiohttp.ClientSession() as session:
        # Прогрев (5 запросов)
        print("Прогрев...")
        warmup_tasks = [send_request(session, semaphore, image_bytes) for _ in range(5)]
        await asyncio.gather(*warmup_tasks)
        
        # Основной тест
        print("Запуск основного теста...")
        start_time = time.perf_counter()
        tasks = [send_request(session, semaphore, image_bytes) for _ in range(TOTAL_REQUESTS)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
    
    # Обработка результатов
    successful_times = [elapsed for success, elapsed in results if success]
    failed = len(results) - len(successful_times)
    
    if not successful_times:
        print("❌ Нет успешных запросов. Проверьте API.")
        return
    
    latencies_ms = [t * 1000 for t in successful_times]
    
    print("\n📊 РЕЗУЛЬТАТЫ")
    print("=" * 40)
    print(f"Успешных:    {len(successful_times)}/{TOTAL_REQUESTS} ({len(successful_times)/TOTAL_REQUESTS*100:.1f}%)")
    print(f"Неуспешных:  {failed}")
    print(f"Общее время: {total_time:.2f} сек")
    print(f"\n⚡ Throughput: {TOTAL_REQUESTS / total_time:.2f} RPS")
    print("\n⏱️  Латентность (мс):")
    print(f"   Avg:  {np.mean(latencies_ms):.2f}")
    print(f"   p50:  {np.percentile(latencies_ms, 50):.2f}")
    print(f"   p95:  {np.percentile(latencies_ms, 95):.2f}")
    print(f"   p99:  {np.percentile(latencies_ms, 99):.2f}")

if __name__ == "__main__":
    asyncio.run(main())