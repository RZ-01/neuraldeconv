import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="mc_sweep_results.csv")
    parser.add_argument("--out",     type=str, default="mc_sweep_plot.pdf")
    args = parser.parse_args()

    df = pd.read_csv(args.results).sort_values("num_pixels")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Pixel Count Sweep  (mc=2000, total pixels fixed)", fontsize=11)

    specs = [
        (axes[0], "psnr", "PSNR (dB)", False, "tab:blue"),
        (axes[1], "ssim", "SSIM",      False, "tab:green"),
        (axes[2], "mse",  "MSE",       True,  "tab:red"),
    ]

    for ax, col, ylabel, logy, color in specs:
        ax.plot(df["num_pixels"], df[col], "o-", color=color, markersize=6, linewidth=1.5)
        for x, y in zip(df["num_pixels"], df[col]):
            ax.annotate(f"{y:.4g}", (x, y),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)

        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Pixels per step (log)", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks(df["num_pixels"])
        ax.set_xticklabels(df["num_pixels"], rotation=45, fontsize=7)
        ax.grid(True, which="both", alpha=0.3, linestyle="--")

        # 顶部 x 轴：steps（同位置，不同标签）
        ax2 = ax.twiny()
        ax2.set_xscale("log")
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(df["num_pixels"])
        ax2.set_xticklabels(df["steps"], rotation=45, fontsize=7)
        ax2.set_xlabel("Steps (log)", fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()