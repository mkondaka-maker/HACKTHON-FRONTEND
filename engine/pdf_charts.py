"""Matplotlib chart builders for the FinSight AI PDF report.

Every builder renders from already-computed engine values into an
in-memory PNG (never screenshots). Callers pass display-ready numbers
with a unit label; builders add titles, legends and captions.
"""

from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

NAVY = "#1F2A44"
CHARCOAL = "#333A45"
ACCENT = "#2F6FED"
ACCENT2 = "#12A594"
ACCENT3 = "#E15728"
ACCENT4 = "#7A5AF8"
GRAY = "#9AA1AD"
GRID = "#E3E7ED"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": NAVY,
    "axes.labelcolor": CHARCOAL,
    "axes.edgecolor": GRAY,
    "xtick.color": CHARCOAL,
    "ytick.color": CHARCOAL,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

_SERIES_COLORS = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#6B7280", "#B54708"]


def _png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def line_chart(years, series, ylabel, title, fmt="{:,.2f}",
               figsize=(7.2, 3.4)):
    """Multi-line time series. series: [(label, [values])]."""
    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, values) in enumerate(series):
        ax.plot(years, values, marker="o", linewidth=2,
                color=_SERIES_COLORS[i % len(_SERIES_COLORS)], label=label)
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(years))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: fmt.format(v)))
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="--", alpha=0.9)
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID, loc="best",
              fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
    return _png(fig)


def grouped_bar(labels, company_vals, peer_vals, ylabel, title,
                company_name="Company", figsize=(7.2, 3.4), fmt="{:,.2f}"):
    """Two-group bars (company vs peer average) across metric labels."""
    import numpy as np

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, company_vals, width, label=company_name,
           color=ACCENT, edgecolor="white")
    ax.bar(x + width / 2, peer_vals, width, label="Peer Average",
           color=GRAY, edgecolor="white")
    ax.set_title(title, loc="left", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: fmt.format(v)))
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID, fontsize=8)
    return _png(fig)


def hbar_factors(labels, scores, title, figsize=(7.2, 3.2)):
    """Horizontal factor bars (0-100) with neutral line at 50."""
    colors = [
        "#B3372F" if s < 40 else "#9A6B0F" if s < 60 else "#1E7F4F"
        for s in scores
    ]
    fig, ax = plt.subplots(figsize=figsize)
    y = list(range(len(labels)))
    ax.barh(y, scores, color=colors, edgecolor="white", height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Factor Score (0-100)")
    ax.set_xlim(0, 100)
    ax.axvline(50, color=GRAY, linestyle="--", linewidth=1.2)
    ax.set_title(title, loc="left", pad=12)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6, linestyle="--")
    for i, s in enumerate(scores):
        ax.text(s + 1.2, i, f"{s:.1f}", va="center", fontsize=8,
                color=CHARCOAL)
    return _png(fig)


def stacked_bar(years, components, ylabel, title, figsize=(7.2, 3.4),
                fmt="{:,.2f}"):
    """Stacked bars. components: [(label, [values])] bottom-up."""
    import numpy as np

    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(years))
    for i, (label, values) in enumerate(components):
        vals = [0 if v is None else v for v in values]
        ax.bar(x, vals, bottom=bottom, label=label,
               color=_SERIES_COLORS[i % len(_SERIES_COLORS)],
               edgecolor="white")
        bottom = bottom + np.array(vals, dtype=float)
    ax.set_title(title, loc="left", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: fmt.format(v)))
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID, fontsize=8)
    return _png(fig)


def anomaly_scatter(records, title, figsize=(7.2, 3.2)):
    """Year vs severity scatter. records: [(year, rank, label)]."""
    rank_color = {3: "#B3372F", 2: "#9A6B0F", 1: "#2F6FED"}
    rank_name = {3: "High", 2: "Medium", 1: "Low"}
    fig, ax = plt.subplots(figsize=figsize)
    for rank in (1, 2, 3):
        pts = [(y, r) for (y, r, _) in records if r == rank]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=110,
                   color=rank_color[rank], label=rank_name[rank],
                   edgecolors="white", linewidths=1, zorder=3)
    for year, rank, label in records:
        ax.text(year, rank + 0.12, label, ha="center", fontsize=7,
                color=CHARCOAL)
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Severity")
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Low", "Medium", "High"])
    ax.set_ylim(0.5, 3.6)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="--", alpha=0.9)
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID, fontsize=8)
    return _png(fig)
