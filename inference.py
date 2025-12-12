import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import re
import os
import random
import logging
from datetime import datetime
import sys
import json

# ==================== ФИКСАЦИЯ SEED ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)
print(f"=== Seed фиксирован: {SEED} ===")

# ==================== КОНФИГУРАЦИЯ ====================
DATA_PATH = "data/test.csv"
MODEL_PATH = "/home/kkupitman/llm_diplom/qwen2.5-math-1.5B"
MODEL_NAME = "qwen2.5-math-1.5B"
OUTPUT_DIR = f"outputs/{MODEL_NAME}"

# Параметры инференса
MAX_NEW_TOKENS = 2000
TEMPERATURE = 0.1
DO_SAMPLE = False

# Классы для инференса
INFERENCE_CLASSES = [
    'separable variables',
    'homogenous 2nd order',
    'inhomogenous 3nd order',
    'polynomial',
    'inhomogenous 2nd order',
    'homogenous 3nd order'
]

# ==================== ЛОГИРОВАНИЕ ====================
def setup_logging(output_dir):
    """Настройка логирования в файл и консоль"""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"inference_{timestamp}.log")
    
    # Очистка существующих обработчиков
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"=== Начало инференса {timestamp} ===")
    logger.info(f"Seed: {SEED}")
    logger.info(f"Модель: {MODEL_PATH}")
    logger.info(f"Выходная директория: {output_dir}")
    return logger

# ==================== ОСНОВНОЙ КОД ====================
def generate_answ(row, model, tokenizer):
    """Генерация ответа моделью"""
    eq = row["equation"]
    message = f"Solve the differential equation: {eq}\ Answer in \\boxed{{<answer>}}"
    
    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    
    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return generated.strip()
    except Exception as e:
        logging.error(f"Ошибка генерации для уравнения '{eq[:50]}...': {e}")
        return ""

def get_answer_without_boxed(model_solution):
    """Извлечение ответа из \\boxed{{}}"""
    if not isinstance(model_solution, str):
        return ""
    
    match = re.search(r'\\\[\s*\\boxed\{(.*?)\}\s*\\\]', model_solution)
    if match:
        return match.group(1)
    return model_solution

def main():
    """Основная функция инференса"""
    # Настройка логирования
    logger = setup_logging(OUTPUT_DIR)
    
    # Создание директорий
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info("Загрузка данных...")
    try:
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Загружено {len(df)} уравнений из {DATA_PATH}")
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        return
    
    # Загрузка модели
    logger.info("Загрузка модели и токенизатора...")
    try:
        if os.path.isdir(MODEL_PATH):
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            logger.error(f"Директория модели не найдена: {MODEL_PATH}")
            return
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.eval()
        logger.info("Модель успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        return
    
    # Генерация ответов
    logger.info("Начало генерации ответов...")
    df['model_solution'] = df.apply(lambda row: generate_answ(row, model, tokenizer), axis=1)
    logger.info("Генерация завершена")
    
    # Извлечение ответов из boxed
    logger.info("Извлечение ответов из \\boxed{}...")
    df['model_answer'] = df['model_solution'].apply(get_answer_without_boxed)
    
    # Сохранение результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f'simpe_inference_{timestamp}.csv')
    df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Результаты сохранены в: {output_file}")
    
    # Сохранение конфигурации эксперимента
    config_data = {
        "timestamp": timestamp,
        "seed": SEED,
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "data_path": DATA_PATH,
        "output_file": output_file,
        "inference_params": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "do_sample": DO_SAMPLE,
            "samples_per_class": SAMPLES_PER_CLASS,
            "classes": INFERENCE_CLASSES
        },
        "total_equations": len(df)
    }
    
    config_file = os.path.join(OUTPUT_DIR, f"inference_config_{timestamp}.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Конфигурация сохранена в: {config_file}")
    
    # Логирование статистики
    empty_answers = df['model_answer'].isna().sum() + (df['model_answer'] == '').sum()
    logger.info(f"Статистика генерации:")
    logger.info(f"  Всего уравнений: {len(df)}")
    logger.info(f"  Пустых ответов: {empty_answers}")
    logger.info(f"  Успешных генераций: {len(df) - empty_answers}")
    
    # Примеры результатов
    logger.info("\n=== Примеры результатов ===")
    for i, row in df.head(3).iterrows():
        logger.info(f"Уравнение {i+1}: {row['equation'][:50]}...")
        logger.info(f"Ответ модели: {row['model_answer'][:100]}...")
        logger.info("-" * 50)
    
    logger.info("=== Инференс успешно завершен ===")

if __name__ == "__main__":
    main()
