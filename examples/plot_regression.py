import json

import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = "sarco"
    data = json.load(open(f"{dataset}_performance.json"))
    data_gb = json.load(open(f"{dataset}_lgbm_performance.json"))

    fig = plt.figure()

    plt.plot(data["sizes"], data["scratch_mae"], label="Trained from scratch")
    plt.plot(data["sizes"], data["pretrain_mae"], label="Pre-Trained")
    plt.plot(data_gb["sizes"], data_gb["scratch_mae"], label="Lightgbm")
    plt.legend()
    plt.xlabel("Data size")
    plt.ylabel("MAE")
    plt.title(f"Mean Absolute Error Comparison (Lower is better) / {dataset}")
    plt.show()
