import pandas as pd
from sacrebleu import sentence_bleu

import re
from sympy import symbols, simplify, Eq, Function, Derivative, Symbol, diff, sympify
from sympy.parsing.latex import parse_latex

import multiprocessing as mp
from time import perf_counter


#-----------------------------------
#----------------Нормализация LaTeX
#-----------------------------------

# x — основная независимая переменная
# C — единый символ для нормализации констант интегрирования
x = symbols("x")
y = Function("y")

# Замена альтернативных обозначений в latex
def normalize_functions(expr: str) -> str:
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
# нормализация переменной
def normalize_variables(sym_expr):
    new_expr = sym_expr
    for s in sym_expr.free_symbols:
        if s.name in ["t", "z"]:
            new_expr = new_expr.subs(s, x)
    return new_expr

#Основная чистка LaTeX
def normalize_latex(expr: str, drop_leading_y_equals: bool = True) -> str:
    expr = str(expr).strip()
    expr = expr.replace(r"\left", "").replace(r"\right", "")
    expr = expr.replace(r"\,", "")
    expr = expr.replace(r"\!", "")
    expr = expr.replace("\u00a0", " ")
    expr = normalize_functions(expr)
    expr = expr.strip().rstrip(" .;")

    if drop_leading_y_equals:
        expr = re.sub(r"^\s*y\s*=\s*", "", expr)

    # Нормализация экспоненты:
    # e^x -> \exp(x)
    # e^{-x} -> \exp(-x)
    # e^{2x} -> \exp(2x)
    expr = re.sub(r'(?<![A-Za-z])e\^\{([^{}]+)\}', r'\\exp(\1)', expr)
    expr = re.sub(r'(?<![A-Za-z])e\^([+\-]?[A-Za-z0-9\\]+)', r'\\exp(\1)', expr)

    return expr


#замена производных latex на функции заглушки (sympy плохо справляется с y'' или (y')^2)
def _preprocess_equation_side_for_latex(side: str) -> str:
    side = normalize_latex(side, drop_leading_y_equals=False)

    replacements = [
        (r"y\s*\^\{\s*\\prime\\prime\\prime\\prime\s*2\s*\}", r"u_{4}^2"),
        (r"y\s*\^\{\s*\\prime\\prime\\prime\s*2\s*\}", r"u_{3}^2"),
        (r"y\s*\^\{\s*\\prime\\prime\s*2\s*\}", r"u_{2}^2"),
        (r"y\s*\^\{\s*\\prime\s*2\s*\}", r"u_{1}^2"),
        (r"y\s*'\s*\^\s*2", r"u_{1}^2"),
        (r"y\s*''\s*\^\s*2", r"u_{2}^2"),
        (r"y\s*'''\s*\^\s*2", r"u_{3}^2"),
        (r"y\s*''''\s*\^\s*2", r"u_{4}^2"),
        (r"y\s*\^\{\s*\\prime\\prime\\prime\\prime\s*\}", r"u_{4}"),
        (r"y\s*\^\{\s*\\prime\\prime\\prime\s*\}", r"u_{3}"),
        (r"y\s*\^\{\s*\\prime\\prime\s*\}", r"u_{2}"),
        (r"y\s*\^\{\s*\\prime\s*\}", r"u_{1}"),
        (r"y\s*''''", r"u_{4}"),
        (r"y\s*'''", r"u_{3}"),
        (r"y\s*''", r"u_{2}"),
        (r"y\s*'", r"u_{1}"),
    ]

    for pattern, replacement in replacements:
        side = re.sub(pattern, replacement, side)

    return side

#-----------------------------------
#----------------Парсинг LaTeX в SymPy
#-----------------------------------
DERIVATIVE_SYMBOL_MAP = {
    Symbol("u_{1}"): Derivative(y(x), x),
    Symbol("u_{2}"): Derivative(y(x), (x, 2)),
    Symbol("u_{3}"): Derivative(y(x), (x, 3)),
    Symbol("u_{4}"): Derivative(y(x), (x, 4)),
    Symbol("y"): y(x),
}

# заменя u1, u2, .... на стандартные проивоздные sympy Derivative(y(x), x) и т.п. 
def _convert_placeholder_symbols(sym_expr):
    converted = sym_expr
    for old_sym, new_expr in DERIVATIVE_SYMBOL_MAP.items():
        converted = converted.subs(old_sym, new_expr)
    converted = normalize_variables(converted)
    return converted

def unwrap_equality(expr):
    # 1. Проверяем: это объект "Уравнение" (Eq)?
    if isinstance(expr, Eq):
        # 2. Если да, возвращаем только то, что СПРАВА от знака равно
        return expr.rhs
        # 3. Если это не уравнение, а просто выражение (например, "x + 1"), 
    # возвращаем его как есть
    return expr

#parsing SOLUTION from latex to synmpy + no \int
def parse_solution_latex(parsed_answer: str):
    if parsed_answer is None:
        return None

    pred_ltx = normalize_latex(parsed_answer, drop_leading_y_equals=True)

    # если модель вернула неопределённый интеграл, это не готовый ответ
    if r"\int" in pred_ltx:
        return None

    sol_expr = parse_latex(pred_ltx)
    sol_expr = unwrap_equality(sol_expr)
    sol_expr = normalize_variables(sol_expr)
    return sol_expr

# на вход подаем latex уравнение, например "y'' + y = 0" или "\\frac{dy}{dx} = x^2"
def parse_equation_latex(equation_raw: str):
    eq = str(equation_raw).strip().rstrip(" .;")
    if "=" not in eq:
        raise ValueError(f"Equation must contain '=': {eq}")

    lhs_str, rhs_str = eq.split("=", 1)
    #из строки latex в символьные выражения
    lhs = parse_latex(_preprocess_equation_side_for_latex(lhs_str.strip()))
    rhs = parse_latex(_preprocess_equation_side_for_latex(rhs_str.strip()))

    # замена символов u_1, u_2… на стандратные sympy производные (см. словарь)
    lhs = _convert_placeholder_symbols(lhs)
    rhs = _convert_placeholder_symbols(rhs)
    return lhs, rhs #возращаем левую и правую части


SYMPY_LOCALS = {
    "x": x,
    "y": y,
    "Eq": Eq,
    "Derivative": Derivative,
    "diff": diff,
}
    
#на вход подаем уравнения в формате sympy
def parse_equation_sympy(equation_sympy: str):
    eq_str = str(equation_sympy).strip().rstrip(" .;")

    if eq_str.startswith("Eq("):
        # превращение в sympy объект 
        eq_obj = sympify(eq_str, locals=SYMPY_LOCALS)
        if not isinstance(eq_obj, Eq):
            raise ValueError(f"Expected Eq(...), got: {eq_str}")
        return eq_obj.lhs, eq_obj.rhs

    if "=" not in eq_str:
        raise ValueError(f"Equation must contain '=': {eq_str}")

    lhs_str, rhs_str = eq_str.split("=", 1)
    lhs = sympify(lhs_str.strip(), locals=SYMPY_LOCALS)
    rhs = sympify(rhs_str.strip(), locals=SYMPY_LOCALS)
    return lhs, rhs

#Автоматическое определения формата уравнения и применяем нужный парасер (sympy или latex)
def parse_equation_auto(equation: str):
    text = str(equation)
    if any(token in text for token in ["Derivative(", "diff(", "Eq(", "y(x)"]):
        return parse_equation_sympy(text)
    return parse_equation_latex(text)


#-----------------------------------
#----------------Проверка эквивалентности решений
#-----------------------------------

# тут проверка simplify(e1 - e2) == 0, т.е. эквивалентность ответов
def reference_equivalence_check(parsed_answer: str, reference_answer: str) -> bool:
    if parsed_answer is None:
        return False

    try:
        ref_ltx = normalize_latex(reference_answer, drop_leading_y_equals=True)
        pred_ltx = normalize_latex(parsed_answer, drop_leading_y_equals=True)

        # если модель вернула неопределённый интеграл, это не готовый ответ
        if r"\int" in pred_ltx:
            return False

        e1 = parse_latex(ref_ltx)
        e2 = parse_latex(pred_ltx)

        e1 = unwrap_equality(e1)
        e2 = unwrap_equality(e2)

        e1 = normalize_variables(e1)
        e2 = normalize_variables(e2)

        return simplify(e1 - e2) == 0

    except Exception as e:
        print("REFERENCE CHECK ERROR")
        print("parsed_answer =", parsed_answer)
        print("reference_answer =", reference_answer)
        print("error =", repr(e))
        return False


#sol_expr - уравнение кандидат, а lsh = rhs - само уравнение
def substitute_solution_and_build_residual(lhs, rhs, sol_expr):
    expr = lhs - rhs
    expr = expr.subs(y(x), sol_expr)
    expr = expr.subs(Derivative(y(x), x), sol_expr.diff(x))
    expr = expr.subs(Derivative(y(x), (x, 2)), sol_expr.diff(x, 2))
    expr = expr.subs(Derivative(y(x), (x, 3)), sol_expr.diff(x, 3))
    expr = expr.subs(Derivative(y(x), (x, 4)), sol_expr.diff(x, 4))
    # принудительное вычисление операций (например производной)
    expr = expr.doit()
    residual = simplify(expr)
    return residual
    
# подстановка решения в ОДУ и сравнение с нулем через sympy
def verify_solution_against_ode(parsed_answer: str, equation: str) -> bool:
    if parsed_answer is None:
        return False

    try:
        # парсим ответ (решение)
        sol_expr = parse_solution_latex(parsed_answer)
        if sol_expr is None:
            return False
        #парсим уравнение
        lhs, rhs = parse_equation_auto(equation)
        #проверка, что solution - решает уравнение
        residual = substitute_solution_and_build_residual(lhs, rhs, sol_expr)

        # residual уже упрощен внутри substitute_solution_and_build_residual
        return residual == 0

    except Exception as e:
        print("ODE CHECK ERROR")
        print("parsed_answer =", parsed_answer)
        print("equation =", equation)
        print("error =", repr(e))
        return False
# проверка на дифф уравение
def _looks_like_equation(text: str) -> bool:
    if text is None:
        return False
    text = str(text)
    derivative_markers = [
        "Derivative(",
        "diff(",
        "y(x)",
        "y'",
        "y''",
        "y'''",
        r"y^{\prime}",
        r"y^{\prime\prime}",
        r"y^{\prime\prime\prime}",
    ]
    return ("=" in text) and any(marker in text for marker in derivative_markers)

    
# вызывает либо verify_solution_against_ode (прверка подстановкой), либо reference_equivalence_check (проверка экивалентности ответов)
def symbolic_check(parsed_answer: str, target: str) -> bool:
    if _looks_like_equation(target):
        return verify_solution_against_ode(parsed_answer, target)
    return reference_equivalence_check(parsed_answer, target)


#-----------------------------------
#----------------Обработка таймаутов (multiprocessing)
#-----------------------------------
# запуск symbolic_check с timeout по времени, чтобы не было долгих sympy вычислений (могут быть зависания)
def _symbolic_worker(parsed_answer, equation, queue):
    try:
        val = symbolic_check(parsed_answer, equation)
        queue.put(("ok", bool(val)))
    except Exception as e:
        queue.put(("error", repr(e)))

def symbolic_check_with_timeout(parsed_answer, equation, timeout_sec=8):
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()

    queue = ctx.Queue()
    proc = ctx.Process(target=_symbolic_worker, args=(parsed_answer, equation, queue))
    proc.start()
    proc.join(timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        # Вычисления длились дольше 8 секунд 
        return pd.NA, "timeout"

    if queue.empty():
        # Процесс завершился, но ничего не прислал
        return False, "worker_no_result"

    status, payload = queue.get()
    if status == "ok":
        return payload, "exact"

    # Во время расчёта произошла ошибка
    return False, f"error:{payload}"

# возращает итог сравнения ответов и причина почему такой результат сравнения

#-----------------------------------
#----------------Вычисление метрик
#-----------------------------------
# NaN/str = "nan"/"None" -> None
def blank_to_none(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    return v

# не вычислеям sympy для ответов, которых не получили
def quick_reject(parsed_answer):
    if parsed_answer is None:
        return True

    txt = str(parsed_answer).strip().lower()
    bad_markers = [
        r"\int",
        "no elementary solution",
        "no simple closed-form solution",
        "reach max function call limit",
        "timed out",
        "timeout_decorator",
    ]
    return any(marker in txt for marker in bad_markers)

# sympy для одной строки с timeout 
def compute_one_row_sympy(row, timeout_sec=8):
    parsed_answer = row["parsed_answer"]
    parse_success = row["parse_success"]
    target_equation = row["equation"]

    if parse_success != 1 or parsed_answer is None:
        return False, "parse_fail"

    if quick_reject(parsed_answer):
        return False, "quick_reject"

    return symbolic_check_with_timeout(parsed_answer, target_equation, timeout_sec=timeout_sec)

# вычисление метрики BLEU (классическая с равными весами) для одной строки
def safe_bleu(prediction: str, reference: str) -> float:
    if prediction is None or str(prediction).strip() == "":
        return 0.0

    return sentence_bleu(
        str(prediction),
        [str(reference)],
        tokenize="char",
        lowercase=False,
        smooth_method="exp"
    ).score / 100.0

def compute_metrics(
    data: pd.DataFrame,
    porog: float = 0.5, # порог по классу (coverage) для проверки успешности по классу
    progress_log_path: str | None = None, # файл, куда писать детальный лог (каждая строка).
    log_every: int = 10, # как часто писать прогресс.
    partial_eval_save_path: str | None = None, # файл для промежуточных сохранений
    save_every: int = 25, # для сохранения частичных результатов
    compute_reference: bool = False, # если True, считает дополнительную метрику reference_equiv_rate (сравнение с эталоном)
    timeout_sec: int = 8,
):
    data = data.copy()
    #для вывода логов сразу же 
    def log_msg(msg: str):
        print(msg, flush=True)
        if progress_log_path is not None:
            with open(progress_log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    # Приводим все none к одному виду
    for col in ["parsed_answer", "llm_solution", "answer", "equation", "equation_sympy"]:
        if col in data.columns:
            data[col] = data[col].apply(blank_to_none)
    #счетчки времени
    total_t0 = perf_counter()
    n = len(data)

    log_msg("=" * 80)
    log_msg(f"compute_metrics started | rows={n} | porog={porog} | timeout_sec={timeout_sec}")

    t0 = perf_counter()

    # sympy проверка для каждого уравнения True/False/NA   
    sympy_vals = []
    # статуст sympy проверки для каждого уравнения "exact", "timeout", "parse_fail" и т.д.
    sympy_statuses = []
    # BLEU для каждого уравнения
    bleu_vals = []
    ref_vals = []

    loop_t0 = perf_counter()

    for i, row in enumerate(data.itertuples(index=False), start=1):
        # строку в словарь
        row = row._asdict()

        eq_preview = str(row["equation"]).replace("\n", " ")
        log_msg(f"[row-start] {i}/{n} | type={row.get('type', 'NA')} | equation={eq_preview}")

        sympy_val, sympy_status = compute_one_row_sympy(
            row=row, 
            timeout_sec=timeout_sec,
        )
        sympy_vals.append(sympy_val)
        sympy_statuses.append(sympy_status)

        if "answer" in row:
            bleu_val = safe_bleu(row.get("parsed_answer"), row["answer"])
            # если нужно было сравнить именно answer модели и изначальный ответ (без подстановки в уравнение)
            if compute_reference:
                if row.get("parsed_answer") is None:
                    ref_val = False
                else:
                    ref_val = reference_equivalence_check(row["parsed_answer"], row["answer"])
            else:
                ref_val = None
        else:
            bleu_val = 0.0
            ref_val = None

        bleu_vals.append(bleu_val)
        ref_vals.append(ref_val)

        log_msg(f"[row-sympy] {i}/{n} | status={sympy_status}")
        log_msg(f"[row-done] {i}/{n}")

        # сохранение прогресса по метрикам
        if partial_eval_save_path is not None and (i % save_every == 0 or i == n):
            tmp = data.iloc[:i].copy()
            tmp["sympy_bool"] = sympy_vals
            tmp["sympy_status"] = sympy_statuses
            tmp["bleu"] = bleu_vals
            if compute_reference:
                tmp["reference_equiv_bool"] = ref_vals
            tmp.to_csv(partial_eval_save_path, index=False)
            log_msg(f"[partial-save] rows saved: 1..{i}")

        # логирование прогресса по строкам и времени
        if i == 1 or i % log_every == 0 or i == n:
            elapsed = perf_counter() - loop_t0
            avg = elapsed / i
            eta = avg * (n - i)
            log_msg(f"[progress] {i}/{n} | elapsed={elapsed:.1f}s | avg/row={avg:.2f}s | eta={eta:.1f}s")

    data["sympy_bool"] = sympy_vals
    data["sympy_status"] = sympy_statuses
    data["bleu"] = bleu_vals
    if compute_reference:
        data["reference_equiv_bool"] = ref_vals
    # перевод в числовые значения метрик, чтобы вычислить метрик по классам 
    # True → 1.0, False → 0.0, а если pd.NA (таймаут) → NaN (пропуск). - мягкая оценка (пропускаем NaN)
    sympy_observed = pd.to_numeric(data["sympy_bool"], errors="coerce")
    # берёт sympy_observed и заменяет NaN на 0.0 (таймаут считается ошибкой). - жесткая оценка (учитываем в среднем NaN)
    sympy_strict = sympy_observed.fillna(0.0)
    # для подсчета доли timeouts
    timeout_mask = data["sympy_status"].astype(str).str.contains("timeout", na=False)
    
    class_observed = (
        data.assign(sympy_num=sympy_observed)
        .groupby("type", dropna=False)["sympy_num"]
        .mean()
        .to_dict()
    )

    class_strict = (
        data.assign(sympy_num=sympy_strict)
        .groupby("type", dropna=False)["sympy_num"]
        .mean()
        .to_dict()
    )

    # количество классов прошедщих порог
    coverage_score_strict = (
        sum(score >= porog for score in class_strict.values()) / len(class_strict)
        if len(class_strict) > 0 else 0.0
    )

    metrics = {
        "parse_success_rate": data["parse_success"].mean(), # доля успешно извлечнных ответов
        "sympy_pass_rate_observed": sympy_observed.mean(), # доля правильных ответов без учеты timeout
        "sympy_pass_rate_strict": sympy_strict.mean(), # доля правильных ответов с учетом timeout
        "sympy_timeout_rate": timeout_mask.mean(), # доля строк, где проверка не уложилась в timeout
        "bleu_mean": pd.to_numeric(data["bleu"], errors="coerce").mean(), 
        f"coverage_score_{porog}": coverage_score_strict,# количество классов прошедщих порог
    }

    # если еще хотели сравнивать именно ответы
    if compute_reference and "reference_equiv_bool" in data.columns:
        metrics["reference_equiv_rate"] = pd.to_numeric(
            data["reference_equiv_bool"], errors="coerce"
        ).mean()

    metrics_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])

    all_types = sorted(set(class_observed.keys()) | set(class_strict.keys()))
    class_df = pd.DataFrame({"equation_type": all_types})
    class_df["class_wise_pass_rate_observed"] = class_df["equation_type"].map(class_observed)
    class_df["class_wise_pass_rate_strict"] = class_df["equation_type"].map(class_strict)

    log_msg(f"[done] compute_metrics total = {perf_counter() - total_t0:.2f}s")
    log_msg("=" * 80)

    return data, metrics_df, class_df
