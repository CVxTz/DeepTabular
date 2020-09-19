import json

import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = json.load(open("census_pretrain_performance.json"))

    fig = plt.figure()

    plt.plot(data["sizes"], data["scratch_accuracies"], label="Trained from scratch")
    plt.plot(data["sizes"], data["pretrain_accuracies"], label="Pre-Trained")
    plt.legend()
    plt.show()
