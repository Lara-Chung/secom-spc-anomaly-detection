"""Plot test-period precision-recall curves and the daily fail-rate trend.

Usage: uv run python src/make_plots.py   (after src/train_evaluate.py)
Reads outputs/run_scores.csv, outputs/metrics.csv, outputs/sql/fail_trends_daily.csv.
Writes outputs/figures/pr_curve_test.png and outputs/figures/fail_rate_daily.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

from secom_data import OUTPUT_DIR
from train_evaluate import METHODS

FIG_DIR = OUTPUT_DIR / "figures"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
# (colour, linestyle, label). The three unsupervised series use a fixed
# categorical palette checked over every colour pair with an OKLab colour-vision
# simulation (dataviz validate_palette.js, --pairs all): worst pair aqua vs
# orange, Delta E 9.2 under deutan, above the 8.0 target. No fourth hue passed
# the same check, so the supervised reference uses neutral ink and a dash-dot
# line instead of a colour of its own.
SERIES = {
    "isolation_forest": ("#2a78d6", "-", "Isolation Forest"),
    "spc_frozen": ("#eb6834", "-", "SPC 3σ, frozen train limits"),
    "spc_rolling": ("#1baf7a", "-", "SPC 3σ, rolling past-only (supplementary)"),
    "logistic_regression": (INK, "-.", "Logistic regression (supervised reference)"),
}
assert set(SERIES) == set(METHODS), "every scored method needs a plot style"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "text.color": INK,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_pr_curves() -> None:
    scores = pd.read_csv(OUTPUT_DIR / "run_scores.csv")
    metrics = pd.read_csv(OUTPUT_DIR / "metrics.csv").set_index(["method", "split"])
    test = scores[scores["split"] == "test"]
    prevalence = test["is_fail"].mean()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_max = 0.0
    for method, (color, linestyle, name) in SERIES.items():
        precision, recall, _ = precision_recall_curve(test["is_fail"], test[method])
        # Drop the (recall 0, precision 1) end point sklearn appends; it is not an observed threshold
        precision, recall = precision[:-1], recall[:-1]
        y_max = max(y_max, precision.max())
        m = metrics.loc[(method, "test")]
        # steps-post on sklearn's decreasing-recall arrays holds precision[i] over
        # (recall[i+1], recall[i]], the same convention as average_precision_score
        # and sklearn's PrecisionRecallDisplay
        ax.step(recall, precision, where="post", color=color, linestyle=linestyle, linewidth=2,
                label=f"{name}  (AP {m['average_precision']:.3f})")
        # Operating point at the train-derived threshold; clip_on=False keeps a
        # point on the axis boundary (e.g. recall 0) fully visible
        ax.plot(m["recall"], m["precision"], "o", markersize=8, color=color,
                markeredgecolor=SURFACE, markeredgewidth=2, clip_on=False, zorder=5)
    ax.axhline(prevalence, color=INK_SECONDARY, linewidth=1.2, linestyle="--",
               label=f"Random ranking (test fail rate {prevalence:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, min(1.0, y_max * 1.2))  # headroom for the legend; nothing is clipped
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Test period precision-recall ({len(test)} runs, {int(test['is_fail'].sum())} fails)",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.text(0.01, 0.01, "Dots: operating point at the threshold chosen on the training period.",
             color=INK_SECONDARY, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG_DIR / "pr_curve_test.png", dpi=150)
    plt.close(fig)


def plot_fail_trend() -> None:
    daily = pd.read_csv(OUTPUT_DIR / "sql" / "fail_trends_daily.csv", parse_dates=["day"])
    scores = pd.read_csv(OUTPUT_DIR / "run_scores.csv", parse_dates=["ts"])
    split_start = scores.loc[scores["split"] == "test", "ts"].min()

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(daily["day"], daily["fail_rate"], width=0.8, color="#86b6ef", label="Daily fail rate")
    ax.plot(daily["day"], daily["fail_rate_trailing_7d"], color="#2a78d6", linewidth=2,
            label="Trailing 7-day fail rate")
    ax.axvline(split_start, color=INK_SECONDARY, linewidth=1.2, linestyle="--")
    ax.text(split_start, ax.get_ylim()[1] * 0.95, "  test period →", color=INK_SECONDARY,
            fontsize=9, va="top")
    ax.set_ylabel("Fail rate")
    ax.set_title("Fail rate by test date (SQL: 02_fail_trends)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fail_rate_daily.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_pr_curves()
    plot_fail_trend()
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
