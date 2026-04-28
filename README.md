# Step-wise Reasoning for Solving ODEs with Small LLMs

## 1. Задача

В работе исследуется применение небольших языковых моделей (LLM)  
для решения обыкновенных дифференциальных уравнений (ODE).

Цель работы — не только получать корректный финальный ответ,  
но и генерировать структурированные пошаговые решения,  
которые можно оценивать и улучшать с помощью **step-wise reward**.

---

## 2. Подход

Работа построена по многоэтапному пайплайну, вдохновлённому статьёй:

**"Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning"**

### Pipeline

1. **Baseline (Qwen2.5-Math-1.5B)**  
   Базовая модель оценивается без дополнительного обучения.

2. **Answer-only SFT**  
   Модель обучается предсказывать только финальный ответ:

   ```text
   equation → \boxed{answer}
   ```

   Используется как контрольная линия.

3. **Reasoning-SFT**  
   Модель обучается генерировать структурированные решения:

   ```text
   equation →
   ACTION 1
   REASON 1
   ...
   FINAL: \boxed{answer}
   ```

   Траектории формируются по шаблонам для каждого типа уравнений.

4. **Построение step-level датасета**  
   Каждое решение разбивается на шаги:

   ```text
   (equation + previous_steps) → next_step
   ```

   Это позволяет оценивать корректность решения на каждом шаге.

5. **Step-wise reward (текущий этап)**

   Для каждого шага вычисляется reward:

   * **Step reward** — совпадение действия (action) с эталонным (0/1)
   * **Final reward** — корректность финального ответа (SymPy)

   Переход:

   ```text
   expert trajectories → step-wise reasoning → reward modeling
   ```

---

## 3. Датасеты

Исходные данные:

* `equation`
* `answer`
* `type` (тип уравнения)

### Построенные датасеты

* **Малые подвыборки (для отладки):**
  * `balanced_train_small`
  * `balanced_val_small`

* **Большие разбиения:**
  * `train_big`
  * `val_big`
  * `balanced_val_metrics_big`

* **Reasoning-датасеты:**
  * содержат `prompt` и `expert_trajectory`

* **Step-датасеты:**
  * пары `previous_steps → target_step`
  * извлечённый `target_action` для reward

---

## 4. Реализованные компоненты

* ✔ Baseline inference и метрики (SymPy)  
* ✔ Answer-only SFT  
* ✔ Reasoning-SFT (на шаблонных траекториях)  
* ✔ Построение step-level датасета  
* ✔ Step-wise reward:
  * exact match по action (0/1)
  * SymPy-проверка финального ответа

Все компоненты реализованы как отдельные скрипты  
и логируются через **MLflow**.

---

## 5. Результаты (малые датасеты)

Сравнение по **SymPy strict accuracy**:

| Тип уравнения          | Baseline | SFT (answer) | SFT (reasoning) |
| ---------------------- | -------- | ------------ | --------------- |
| homogenous 2nd order   | 1.00     | 0.92         | 0.94            |
| homogenous 3rd order   | 0.95     | 0.80         | 0.80            |
| inhomogenous 2nd order | 0.05     | 0.14         | 0.06            |
| inhomogenous 3rd order | 0.00     | 0.04         | 0.00            |
| polynomial             | 1.00     | 1.00         | 0.76            |
| separable variables    | 0.00     | 0.06         | 0.02            |

---

## 6. Интерпретация результатов

* Answer-only SFT улучшает финальную точность (SymPy),  
  но не решает проблему сложных классов.

* Модель хорошо работает на:
  * polynomial
  * homogeneous уравнениях

* Модель плохо работает на:
  * inhomogeneous
  * separable variables

* Reasoning-SFT не всегда повышает финальную точность,  
  но меняет структуру решения (появляются шаги). При этом модель начала решать новые типы уравнений

* Это показывает, что обучение только на ответах  
  не формирует стратегию решения.

---

## 7. Условия экспериментов

* Используются **малые подвыборки (small splits)**  
  для отладки пайплайна.

* Класс `any` исключён из обучения.

* Baseline — без обучения  
* Answer-SFT — обучение на `\boxed{answer}`  
* Reasoning-SFT — обучение на `expert_trajectory`

* Используется **лучшая эпоха по SymPy strict**

---

## 8. Текущий статус

* ✔ Reasoning-SFT реализован  
* ✔ Step-wise reward реализован  

---

## 9. Запуск

```bash
python scripts/make_val_train_sets.py
python scripts/reasoning_dataset_builder.py
python scripts/run_sft_reasoning_big.py
python scripts/step_dataset_builder.py
python scripts/run_step_rewards_exact.py
```

