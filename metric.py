import re
from sympy import symbols, simplify
from sympy.parsing.latex import parse_latex
from sacrebleu import sentence_bleu
import pandas as pd
import numpy as np
import random
import torch
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
os.environ['PYTHONHASHSEED'] = str(SEED)
print(f"=== Seed фиксирован: {SEED} ===")

# ==================== КОНФИГУРАЦИЯ ====================
PATH_TO_DATA_FRAME = "/home/ksenia/Desktop/lab/exper_artical/outputs/qwen2.5-math-1.5B/simpe_inference.csv"
OUTPUT_DIR = os.path.dirname(PATH_TO_DATA_FRAME)

# Параметры метрик
COVERAGE_THRESHOLDS = [0.5, 0.7, 0.9]
CLASSES_FOR_METRICS = [
    'separable variables',
    'homogenous 2nd order',
    'inhomogenous 3nd order',
    'polynomial',
    'inhomogenous 2nd order',
    'homogenous 3nd order'
]

# SymPy символы
x, C = symbols('x C')

# ==================== ЛОГИРОВАНИЕ ====================
def setup_logging(output_dir):
    """Настройка логирования"""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"metrics_{timestamp}.log")
    
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
    logger.info(f"=== Начало расчета метрик {timestamp} ===")
    logger.info(f"Seed: {SEED}")
    logger.info(f"Данные: {PATH_TO_DATA_FRAME}")
    return logger

# ==================== ВАША ЛОГИКА МЕТРИК ====================
def normalize_functions(expr: str) -> str:
    """Нормализация названий функций"""
    replacements = {
        r"\\tg": r"\\tan",
        r"\\ctg": r"\\cot",
        r"\\cotan": r"\\cot",
        r"\\ctan": r"\\cot",
        r"\\arctg": r"\\atan",
        r"\\arcctg": r"\\acot",
        r"\\ln": r"\\log",
        r"\\arcsin": r"\\asin",
        r"\\arccos": r"\\acos",
        r"\\sh": r"\\sinh",
        r"\\ch": r"\\cosh",
        r"\\th": r"\\tanh",
    }
    for old, new in replacements.items():
        expr = re.sub(old, new, expr)
    return expr

def normalize_latex(expr: str) -> str:
    """Нормализация LaTeX выражений"""
    if not isinstance(expr, str):
        return ""
    
    expr = expr.strip()
    expr = expr.replace("y=", "")
    expr = re.sub(r"\\boxed\{(.*?)\}", r"\1", expr)
    expr = expr.replace(r"\left", "").replace(r"\right", "")
    expr = normalize_functions(expr)
    return expr

def normalize_variables(sym_expr):
    """Приводим t, y, z к x"""
    new_expr = sym_expr
    for s in sym_expr.free_symbols:
        if s.name in ['t', 'y', 'z']:
            new_expr = new_expr.subs(s, x)
    return new_expr

def normalize_constants(sym_expr):
    """Нормализация констант"""
    new_expr = sym_expr
    for s in sym_expr.free_symbols:
        if s != x and s.name not in ['t', 'y', 'z']:
            new_expr = new_expr.subs(s, C)
    return simplify(new_expr)

def sympy_pass_rate(real_answer, model_answer, verbose=False):
    """Проверка совпадения ответов через SymPy"""
    try:
        real_expr_ltx = normalize_latex(real_answer)
        model_expr_ltx = normalize_latex(model_answer)

        if "\\int" in model_expr_ltx:
            return False

        e1 = parse_latex(real_expr_ltx)
        e2 = parse_latex(model_expr_ltx)

        e1 = normalize_variables(e1)
        e2 = normalize_variables(e2)

        e1 = normalize_constants(e1)
        e2 = normalize_constants(e2)

        diff_expr = simplify(e1 - e2)
        eq = diff_expr == 0

        return eq
    except Exception as e:
        if verbose:
            logging.debug(f"SymPy error: {e}")
        return False

def class_wise_pass_rate(df, type_eq):
    """Метрика pass rate по классам"""
    num_correct = df[(df["type"] == type_eq) & (df["sympy_bool"])].shape[0]
    total_eqs = df[df["type"] == type_eq].shape[0]
    if total_eqs == 0:
        return 0.0
    return num_correct / total_eqs

def coverage_score(threshold, d):
    """Coverage score для порога"""
    s = sum([1 for met in d.values() if met > threshold])
    return s/len(d) if d else 0.0

def bleu_score(reference, hypothesis):
    """Вычисление BLEU score"""
    if not isinstance(hypothesis, str) or not isinstance(reference, str):
        return 0.0
    
    try:
        return sentence_bleu(
            hypothesis, 
            [reference], 
            tokenize='char',
            lowercase=False,
            smooth_method='exp'
        ).score / 100
    except:
        return 0.0

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def calculate_metrics():
    """Основной расчет метрик"""
    logger = setup_logging(OUTPUT_DIR)
    
    logger.info("Загрузка данных...")
    try:
        df = pd.read_csv(PATH_TO_DATA_FRAME)
        logger.info(f"Загружено {len(df)} записей")
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        return
    
    # Расчет метрик для каждого уравнения
    logger.info("Расчет SymPy pass rate...")
    df["sympy_bool"] = df.apply(
        lambda row: sympy_pass_rate(row["answer"], row["model_answer"]), 
        axis=1
    )
    
    logger.info("Расчет BLEU score...")
    df["bleu"] = df.apply(
        lambda row: bleu_score(row["answer"], row["model_answer"]), 
        axis=1
    )
    
    # Сохранение первого файла с метриками
    base_name = PATH_TO_DATA_FRAME[:-4]  # убираем ".csv"
    output_path_1 = base_name + "_sympy_pass_rate_and_bleu.csv"
    df.to_csv(output_path_1, index=False, encoding='utf-8')
    logger.info(f"Детальные метрики сохранены в: {output_path_1}")
    
    # Метрики по классам
    logger.info("Расчет class-wise метрик...")
    d_met = {}
    errors_by_class = {}
    
    for t in CLASSES_FOR_METRICS:
        class_df = df[df["type"] == t]
        if len(class_df) > 0:
            pass_rate = class_wise_pass_rate(df, t)
            d_met[t] = pass_rate
            errors = class_df[~class_df["sympy_bool"]]
            errors_by_class[t] = len(errors)
            logger.info(f"Класс '{t}': Pass rate = {pass_rate:.2%}, Ошибок = {len(errors)}")
        else:
            d_met[t] = 0.0
            errors_by_class[t] = 0
            logger.warning(f"Класс '{t}': Уравнения не найдены")
    
    # Coverage Score
    logger.info("Расчет Coverage Score...")
    d_met["Coverage_Score_0_5"] = coverage_score(0.5, d_met)
    d_met["Coverage_Score_0_7"] = coverage_score(0.7, d_met)
    d_met["Coverage_Score_0_9"] = coverage_score(0.9, d_met)
    
    # Сохранение второго файла с метриками по классам
    output_path_2 = base_name + "_class_wise_pass_rate_and_coverage_score.csv"
    metrics_df = pd.DataFrame(list(d_met.items()), columns=['Equation_Type', 'Score'])
    metrics_df.to_csv(output_path_2, index=False, encoding='utf-8')
    logger.info(f"Метрики по классам сохранены в: {output_path_2}")
    
    # Сохранение ошибок
    errors_df = df[~df["sympy_bool"]]
    if len(errors_df) > 0:
        errors_file = base_name + "_errors.csv"
        errors_df.to_csv(errors_file, index=False, encoding='utf-8')
        logger.info(f"Сохранено {len(errors_df)} ошибок в: {errors_file}")
    
    # Сохранение сводки
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "seed": SEED,
        "total_equations": len(df),
        "sympy_correct": int(df["sympy_bool"].sum()),
        "sympy_pass_rate": float(df["sympy_bool"].mean()),
        "avg_bleu": float(df["bleu"].mean()),
        "median_bleu": float(df["bleu"].median()),
        "class_wise_pass_rates": d_met,
        "errors_by_class": errors_by_class,
        "coverage_thresholds": COVERAGE_THRESHOLDS,
        "classes_analyzed": CLASSES_FOR_METRICS
    }
    
    summary_file = os.path.join(OUTPUT_DIR, f"metrics_summary_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Сводка метрик сохранена в: {summary_file}")
    
    # Вывод результатов в лог
    logger.info("\n" + "="*60)
    logger.info("ИТОГОВЫЕ МЕТРИКИ:")
    logger.info(f"Всего уравнений: {len(df)}")
    logger.info(f"Правильных ответов (SymPy): {df['sympy_bool'].sum()} ({df['sympy_bool'].mean():.2%})")
    logger.info(f"Средний BLEU: {df['bleu'].mean():.4f}")
    logger.info(f"Coverage Score (0.5): {d_met['Coverage_Score_0_5']:.2%}")
    logger.info(f"Coverage Score (0.7): {d_met['Coverage_Score_0_7']:.2%}")
    logger.info(f"Coverage Score (0.9): {d_met['Coverage_Score_0_9']:.2%}")
    
    logger.info("\nМетрики по классам:")
    for cls in CLASSES_FOR_METRICS:
        if cls in d_met:
            logger.info(f"  {cls}: {d_met[cls]:.2%}")
    
    logger.info("\n=== Расчет метрик успешно завершен ===")
    
    return df, d_met

if __name__ == "__main__":
    calculate_metrics()
