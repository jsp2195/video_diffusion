import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_training_curves(history: Dict[str, List[float]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    plt.figure(figsize=(10, 6))
    if history.get("train_loss"):
        plt.plot(epochs, history["train_loss"], label="train_loss")
    if history.get("val_loss"):
        plt.plot(epochs, history["val_loss"], label="val_loss")
    if history.get("fvd_proxy"):
        fvd_epochs = history.get("fvd_proxy_epochs", list(range(1, len(history["fvd_proxy"]) + 1)))
        plt.plot(fvd_epochs, history["fvd_proxy"], label="fvd_proxy")
    elif history.get("fvd"):
        fvd_epochs = history.get("fvd_epochs", list(range(1, len(history["fvd"]) + 1)))
        plt.plot(fvd_epochs, history["fvd"], label="fvd")

    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()
