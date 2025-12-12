import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch.nn.functional as F
import numpy as np
import random
import os
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
DATA_PATH = "data/train.csv"
MODEL_PATH = "qwen2.5-math-1.5B"
OUTPUT_DIR = "outputs/finetuned_model"

# Параметры обучения
NUM_EPOCHS = 8
LEARNING_RATE = 3e-5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
TRAIN_RATIO = 0.9
VAL_RATIO = 0.1

# Классы для обучения
TRAINING_CLASSES = [
    'separable variables',
    'homogenous 2nd order',
    'inhomogenous 3nd order',
    'polynomial',
    'inhomogenous 2nd order',
    'homogenous 3nd order'
]
SAMPLES_PER_CLASS_TRAIN = 6369  # для балансировки классов

# Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== ЛОГИРОВАНИЕ ====================
def setup_logging(output_dir):
    """Настройка логирования"""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
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
    return logger

# ==================== ДАТАСЕТ ====================
class DifferentialEquationDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.data_path = data_path
        
        df = pd.read_csv(data_path)
        
        # Создание сбалансированного датасета
        def get_some_eq(df, ls_type, samples_per_class=2):
            new_df = pd.DataFrame()
            for type_name in ls_type:
                eqs = df[df["type"] == type_name].iloc[:samples_per_class]
                new_df = pd.concat([new_df, eqs])
            return new_df
        
        df = get_some_eq(df, TRAINING_CLASSES, SAMPLES_PER_CLASS_TRAIN)
        
        # Сохраняем данные
        self.equations = df["equation"].tolist()
        self.answers = df["answer"].tolist()
        
        logging.info(f"Датасет создан, размер: {len(self.equations)}")
    
    def __len__(self):
        return len(self.equations)
    
    def __getitem__(self, idx):
        """Токенизация одного уравнения"""
        eq = self.equations[idx]
        answer = self.answers[idx]
        
        # Форматируем промпт
        message = f"Solve the differential equation: {eq} Answer in \\boxed{{<answer>}}"
        
        # Токенизация всего текста
        text = message + answer
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",  # Добавляем паддинг
            return_tensors="pt"
        )
        
        # Создаем labels: маскируем промпт
        labels = encoding["input_ids"].clone()
        prompt_length = len(self.tokenizer(message, truncation=True, max_length=512)["input_ids"])
        labels[:, :prompt_length] = -100  # Маскируем промпт
        
        return {
            'input_ids': encoding["input_ids"].squeeze(0),
            'labels': labels.squeeze(0),
            'attention_mask': encoding["attention_mask"].squeeze(0)
        }

# ==================== ОБУЧЕНИЕ ====================
def train_model(train_set, val_set, model_path, output_dir, tokenizer):
    """Функция обучения модели"""
    logger = setup_logging(output_dir)
    logger.info(f"=== Начало обучения {datetime.now().strftime('%Y%m%d_%H%M%S')} ===")
    logger.info(f"Seed: {SEED}")
    logger.info(f"Модель: {model_path}")
    logger.info(f"Данные: {DATA_PATH}")
    logger.info(f"Выходная директория: {output_dir}")
    
    # Загрузка модели
    logger.info("Загрузка модели...")
    try:
        # Загружаем модель в float32 для стабильности
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Используем float32 для стабильности
            local_files_only=True
        )
        
        # Перемещаем модель на устройство
        model.to(device)
        
        # Устанавливаем pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        
        logger.info(f"Модель успешно загружена на устройство: {device}")
        logger.info(f"Тип данных модели: {model.dtype}")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        return None
    
    # Проверяем поддержку bf16
    use_bf16 = False
    if torch.cuda.is_available():
        try:
            # Проверяем поддержку BF16
            if torch.cuda.is_bf16_supported():
                use_bf16 = True
                logger.info("Поддержка BF16 обнаружена, будет использоваться BF16")
            else:
                logger.info("BF16 не поддерживается, будет использоваться FP32")
        except:
            logger.info("Не удалось проверить поддержку BF16, используется FP32")
    
    # Параметры обучения
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=False,  # Отключаем FP16, используем только BF16 или FP32
        optim="adamw_torch",
        report_to="none",
        logging_dir=os.path.join(output_dir, "logs"),
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
        group_by_length=False,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        push_to_hub=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=False,  # Отключаем для стабильности
    )
    
    # Trainer
    logger.info("Создание Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
    )
    
    # Обучение
    logger.info("Начало обучения...")
    try:
        train_result = trainer.train()
        logger.info("Обучение завершено успешно")
        
        # Сохранение модели
        model_save_path = os.path.join(output_dir, "best_model")
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        logger.info(f"Модель сохранена в: {model_save_path}")
        
        # Сохранение метрик обучения
        if hasattr(trainer, 'state') and trainer.state.log_history:
            metrics_file = os.path.join(output_dir, "logs", "training_metrics.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Метрики обучения сохранены в: {metrics_file}")
        
        # Сохранение конфигурации обучения
        config_data = {
            "timestamp": datetime.now().isoformat(),
            "seed": SEED,
            "model_path": model_path,
            "output_dir": output_dir,
            "training_params": {
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "samples_per_class": SAMPLES_PER_CLASS_TRAIN,
                "classes": TRAINING_CLASSES,
                "precision": "bf16" if use_bf16 else "fp32"
            },
            "dataset_size": len(train_set) + len(val_set),
            "train_size": len(train_set),
            "val_size": len(val_set)
        }
        
        config_file = os.path.join(output_dir, "training_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Конфигурация обучения сохранена в: {config_file}")
        
        # Вывод сводки обучения
        logger.info("\n" + "="*60)
        logger.info("СВОДКА ОБУЧЕНИЯ:")
        logger.info(f"Размер тренировочного датасета: {len(train_set)}")
        logger.info(f"Размер валидационного датасета: {len(val_set)}")
        logger.info(f"Количество эпох: {NUM_EPOCHS}")
        logger.info(f"Learning rate: {LEARNING_RATE}")
        logger.info(f"Batch size: {BATCH_SIZE}")
        logger.info(f"Используемая точность: {'BF16' if use_bf16 else 'FP32'}")
        logger.info(f"Директория модели: {model_save_path}")
        logger.info("="*60)
        
        return model_save_path
        
    except Exception as e:
        logger.error(f"Ошибка обучения: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Основная функция обучения"""
    # Создание директорий
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Настройка логирования
    logger = setup_logging(OUTPUT_DIR)
    logger.info(f"=== Начало обучения {datetime.now().strftime('%Y%m%d_%H%M%S')} ===")
    logger.info(f"Seed: {SEED}")
    logger.info(f"Модель: {MODEL_PATH}")
    logger.info(f"Данные: {DATA_PATH}")
    logger.info(f"Выходная директория: {OUTPUT_DIR}")
    
    # Загрузка токенизатора
    logger.info("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # Установка pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Установлен pad_token_id: {tokenizer.pad_token_id}")
    
    # Создание датасета
    logger.info("Создание датасета...")
    dataset = DifferentialEquationDataset(DATA_PATH, tokenizer)
    
    if len(dataset) == 0:
        logger.error("Датасет пуст!")
        return
    
    # Разделение на train/val
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    
    train_set, val_set = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    logger.info(f"Разделение данных: train={len(train_set)}, val={len(val_set)}")
    
    # Обучение
    model_path = train_model(train_set, val_set, MODEL_PATH, OUTPUT_DIR, tokenizer)
    
    if model_path:
        print(f"\n=== Обучение успешно завершено! ===")
        print(f"Модель сохранена в: {model_path}")
        print(f"Логи сохранены в: {os.path.join(OUTPUT_DIR, 'logs')}")
    else:
        print("\n=== Обучение завершилось с ошибкой! ===")
        print(f"Проверьте логи в: {os.path.join(OUTPUT_DIR, 'logs')}")

if __name__ == "__main__":
    main()
