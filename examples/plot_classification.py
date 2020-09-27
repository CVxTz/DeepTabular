import json

import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = "cover"
    data = json.load(open(f"{dataset}_pretrain_performance.json"))

    fig = plt.figure()

    plt.plot(data["sizes"], data["scratch_accuracies"], label="Trained from scratch")
    plt.plot(data["sizes"], data["pretrain_accuracies"], label="Pre-Trained")
    plt.legend()
    plt.xlabel("Data size")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Comparison (Higher is better) / {dataset}")
    plt.show()
