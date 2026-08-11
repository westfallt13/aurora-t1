import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_plot(metrics_path="metrics.csv", output_path="../training_progress.png"):
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    with open(metrics_path) as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            loss = float(row["loss"])
            if row["split"] == "train":
                train_steps.append(step)
                train_loss.append(loss)
            else:
                eval_steps.append(step)
                eval_loss.append(loss)

    plt.figure(figsize=(9, 5))
    plt.plot(train_steps, train_loss, label="train", alpha=0.8)
    plt.plot(eval_steps, eval_loss, label="eval (held-out)", marker="o", markersize=3)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("aurora-t1 local training progress")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return len(train_steps), len(eval_steps)


if __name__ == "__main__":
    n_train, n_eval = generate_plot()
    print(f"Saved training_progress.png ({n_train} train points, {n_eval} eval points)")
