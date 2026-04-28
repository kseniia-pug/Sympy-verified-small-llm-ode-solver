import json
from pathlib import Path
import pandas as pd

def main():

    CONFIG = {
        "seed": 42,
        "n_debug_per_class": 20, # size of debaging data
        "n_val_per_class": 50, #size of val data
        "n_train_per_class": 300, #size of train data
        "n_mini_test_per_class": 20, #size of test data
    }

    TRAIN_PATH = Path('data/train.csv')
    TEST_PATH = Path('data/test.csv')
    #dir for mini balanced split data
    OUT_DIR = Path('data/subsets')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data_test = pd.read_csv(TEST_PATH, sep=",")
    data_train = pd.read_csv(TRAIN_PATH, sep=",")

    # make debug data
    mini_debug_data = data_train.groupby('type').sample(
        n=CONFIG["n_debug_per_class"], 
        random_state=CONFIG["seed"]
    )
    #eliminate intersection with debug data
    remaining_after_debug = data_train[~data_train.index.isin(mini_debug_data.index)]
    #make mini train data
    balanced_train_small = remaining_after_debug.groupby('type').sample(
        n=CONFIG["n_train_per_class"], 
        random_state=CONFIG["seed"]
    )
    #eliminate intersection with debug data and train data
    remaining_after_debug_and_train = remaining_after_debug[~remaining_after_debug.index.isin(balanced_train_small.index)]
    #make mini val data
    balanced_val_small = remaining_after_debug_and_train.groupby('type').sample(
        n=CONFIG["n_val_per_class"], 
        random_state=CONFIG["seed"]
    )
    #make mini test data
    mini_test_smoke = data_test.groupby('type').sample(
        n=CONFIG["n_mini_test_per_class"], 
        random_state=CONFIG["seed"]
    )

    #saving data
    mini_debug_data.to_csv(OUT_DIR / "mini_debug.csv", index=False)
    print(f"Mini debug data saved to {OUT_DIR}/mini_debug.csv")
    balanced_val_small.to_csv(OUT_DIR / "balanced_val_small.csv", index=False)
    print(f"Small balanced val data saved to {OUT_DIR}/balanced_val_small.csv")
    balanced_train_small.to_csv(OUT_DIR / "balanced_train_small.csv", index=False)
    print(f"Small balanced train data saved to {OUT_DIR}/balanced_train_small.csv")
    mini_test_smoke.to_csv(OUT_DIR / "mini_test.csv", index = False)
    print(f"Mini test data saved to {OUT_DIR}/mini_test.csv")


    #saving config of mini data
    manifest = {
        "mini_debug": {
            "size": len(mini_debug_data),
            "distribution": mini_debug_data["type"].value_counts().sort_index().to_dict(),
            "purpose": "Отладка пайплайна: prompt, extraction, verifier, reward, быстрые sanity-checks."
        },
        "balanced_train_small": {
            "size": len(balanced_train_small),
            "distribution": balanced_train_small["type"].value_counts().sort_index().to_dict(),
            "purpose": "Основной маленький train-набор для LoRA и первых PPO-экспериментов."
        },
        "balanced_val_small": {
            "size": len(balanced_val_small),
            "distribution": balanced_val_small["type"].value_counts().sort_index().to_dict(),
            "purpose": "Фиксированная валидация для сравнения baseline / LoRA / PPO."
        },
        "mini_test_smoke": {
            "size": len(mini_test_smoke),
            "distribution": mini_test_smoke["type"].value_counts().sort_index().to_dict(),
            "purpose": "Быстрая smoke-проверка на test; не использовать для выбора лучшей модели."
        }
    }
    
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {OUT_DIR / 'manifest.json'}")

if __name__ == "__main__":
    main()
