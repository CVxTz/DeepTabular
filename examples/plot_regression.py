import json

import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = json.load(open("news_performance.json"))

    fig = plt.figure()

    plt.plot(data["sizes"], data["scratch_mae"], label="Trained from scratch")
    plt.plot(data["sizes"], data["pretrain_mae"], label="Pre-Trained")
    plt.legend()
    plt.xlabel("Data size")
    plt.ylabel("MAE")
    plt.show()
