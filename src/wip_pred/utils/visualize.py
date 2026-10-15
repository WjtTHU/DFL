import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_true_vs_pred(true_vals, data: list[dict], title: str, save_path: str):
    plt.figure(figsize=(6, 6))
    plt.plot(
        [true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], "r--"
    )  # reference line: y=x
    plt.plot(
        [true_vals.min(), true_vals.max()], [true_vals.min() + 7, true_vals.max() + 7], "r:"
    )  # reference line: y=x+7
    plt.plot(
        [true_vals.min(), true_vals.max()], [true_vals.min() - 7, true_vals.max() - 7], "r:"
    )  # reference line: y=x-7

    palette = sns.color_palette("Set1")
    palette = [palette[1]] + [palette[0]] + palette[2:]
    sns.set_palette(palette)

    prop_text = ""

    for cfg in data:
        if cfg["alpha"] > 1e-3:
            sns.scatterplot(**cfg)
        diff = np.abs(cfg["y"] - cfg["x"]) <= 7
        prop = np.sum(diff) / len(diff)
        mae = np.mean(np.abs(cfg["y"] - cfg["x"]))
        prop_text += f"{cfg['label']:^15s} Prop within ±7: {prop:>5.1%}  MAE: {mae:>5.2f}  Num: {len(cfg['x']):>4}\n"

    prop_text = prop_text.rstrip("\n")

    plt.text(
        x=0.03,  # x-position (3% from the left, relative coordinate)
        y=0.9,  # y-position (90% from the bottom, relative coordinate)
        s=prop_text,
        transform=plt.gca().transAxes,  # Use relative coordinates (independent of data range)
        fontsize=9,
        family="Consolas",
        bbox=dict(facecolor="white", edgecolor="gray", pad=5, alpha=0.3),
    )

    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=300)
    plt.close()
