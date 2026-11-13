from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import config
from .data import ShiftedDataset, apply_shift
from .stats import paired_arrays

COLORS = {"H0": "#2a78d6", "H1": "#eb6834"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
DPI = 300
SHIFT_LABELS = {
    "ruido_gaussiano": "Ruído gaussiano",
    "desfoque_gaussiano": "Desfoque gaussiano",
    "cinza_baixo_contraste": "Cinza + baixo contraste",
    "esboco": "Esboço sintético",
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def _br(value: float, fmt: str, math: bool = True) -> str:
    return format(value, fmt).replace(".", "{,}" if math else ",")


def _format_p(p: float) -> str:
    if p < 1e-3:
        mantissa, exponent = f"{p:.2e}".split("e")
        return rf"$p = {_br(float(mantissa), '.2f')} \times 10^{{{int(exponent)}}}$"
    return f"$p = {_br(p, '.4f')}$"


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path)
    return path


def fig_shift_examples(data: ShiftedDataset, path: Path, per_shift: int = 4) -> Path:
    rng = np.random.default_rng(config.SHIFT_ASSIGNMENT_SEED)
    fig, axes = plt.subplots(len(config.SHIFTS) + 1, per_shift, figsize=(3.5, 4.6))
    picks = rng.choice(len(data.labels), per_shift, replace=False)
    for col, idx in enumerate(picks):
        original = torch.from_numpy(data.images[idx]).permute(2, 0, 1).float() / 255.0
        rows = [("Original", original)] + [
            (SHIFT_LABELS[s], apply_shift(original, s, int(idx))) for s in config.SHIFTS
        ]
        for row, (label, img) in enumerate(rows):
            ax = axes[row, col]
            ax.imshow(img.permute(1, 2, 0).numpy(), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(label, fontsize=6.5, rotation=0, ha="right", va="center")
            if row == 0:
                ax.set_title(data.classnames[data.labels[idx]], fontsize=7)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    return _save(fig, path)


def fig_accuracy_box(df: pd.DataFrame, summary: dict, path: Path) -> Path:
    h0, h1 = paired_arrays(df, "acuracia_top1")
    values = {"H0": 100 * h0, "H1": 100 * h1}
    test = summary["metricas"]["acuracia_top1"]
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    rng = np.random.default_rng(0)
    for pos, scenario in enumerate(config.SCENARIOS, start=1):
        y = values[scenario]
        ax.boxplot(
            y, positions=[pos], widths=0.5, showfliers=False, patch_artist=True,
            boxprops={"facecolor": COLORS[scenario], "alpha": 0.18, "edgecolor": COLORS[scenario], "linewidth": 1.2},
            whiskerprops={"color": COLORS[scenario], "linewidth": 1.2},
            capprops={"color": COLORS[scenario], "linewidth": 1.2},
            medianprops={"color": COLORS[scenario], "linewidth": 2},
        )
        jitter = rng.uniform(-0.13, 0.13, len(y))
        ax.scatter(pos + jitter, y, s=14, color=COLORS[scenario], edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(pos, y.mean(), marker="D", s=34, color="white", edgecolor=INK, linewidth=1.1, zorder=4)
    ax.set_xticks([1, 2], [config.SCENARIO_LABELS[s] for s in config.SCENARIOS])
    ax.set_xlim(0.4, 2.6)
    ax.set_ylabel("Acurácia Top-1 (%)")
    ax.set_title(f"{test['teste']}: {_format_p(test['p_valor'])}, $d_z = {_br(test['cohen_d_pareado'], '.2f')}$")
    handles = [
        plt.Line2D([], [], color=INK_SECONDARY, linewidth=2, label="Mediana"),
        plt.Line2D([], [], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor=INK, label="Média"),
        plt.Line2D([], [], marker="o", linestyle="none", color=INK_SECONDARY, markersize=4, label=f"Rodada (n = {len(h0)})"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False,
              handletextpad=0.4, columnspacing=1.2)
    return _save(fig, path)


def fig_paired_evolution(df: pd.DataFrame, summary: dict, path: Path) -> Path:
    h0, h1 = paired_arrays(df, "acuracia_top1")
    seeds = np.sort(df["seed"].unique())
    diff = 100 * (h1 - h0)
    test = summary["metricas"]["acuracia_top1"]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.16, 3.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.22}
    )
    top.vlines(seeds, 100 * np.minimum(h0, h1), 100 * np.maximum(h0, h1), color=GRID, linewidth=2, zorder=1)
    for scenario, y in (("H0", h0), ("H1", h1)):
        top.plot(seeds, 100 * y, color=COLORS[scenario], linewidth=2, marker="o", markersize=4,
                 markeredgecolor="white", markeredgewidth=0.6, label=config.SCENARIO_LABELS[scenario], zorder=2)
    top.set_ylabel("Acurácia Top-1 (%)")
    top.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, borderaxespad=0.2)
    top.set_title(
        f"H1 supera H0 em {test['rodadas_H1_maior']} de {test['n_pares']} rodadas pareadas "
        f"(empates: {test['rodadas_empate']})",
        loc="left",
    )
    bar_colors = [COLORS["H1"] if d > 0 else COLORS["H0"] if d < 0 else INK_SECONDARY for d in diff]
    bottom.bar(seeds, diff, width=0.7, color=bar_colors)
    bottom.axhline(0, color=INK_SECONDARY, linewidth=0.8)
    bottom.axhline(diff.mean(), color=INK, linewidth=1, linestyle="--")
    bottom.text(0.0, 1.03, f"Tracejado: média de $\\Delta$ = {_br(diff.mean(), '+.2f', math=False)} p.p.", transform=bottom.transAxes,
                ha="left", va="bottom", fontsize=7, color=INK)
    bottom.set_ylabel(r"$\Delta$ (H1 $-$ H0, p.p.)")
    bottom.set_xlabel("Rodada (semente)")
    bottom.set_xticks(seeds)
    bottom.set_xlim(seeds[0] - 0.6, seeds[-1] + 0.6)
    return _save(fig, path)


def fig_metric_bars(summary: dict, path: Path) -> Path:
    metrics = (("acuracia_top1", "Acurácia Top-1"), ("f1_macro", "F1-Score Macro"))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    width = 0.36
    gap = 0.02
    x = np.arange(len(metrics))
    for offset, scenario in ((-1, "H0"), (1, "H1")):
        means = [100 * summary["metricas"][m][scenario]["media"] for m, _ in metrics]
        sds = [100 * summary["metricas"][m][scenario]["desvio_padrao"] for m, _ in metrics]
        positions = x + offset * (width + gap) / 2
        ax.bar(positions, means, width=width, color=COLORS[scenario], label=config.SCENARIO_LABELS[scenario],
               yerr=sds, capsize=3, error_kw={"elinewidth": 1, "ecolor": INK, "capthick": 1})
        for px, m, s in zip(positions, means, sds):
            ax.annotate(_br(m, ".1f", math=False), xy=(px, m + s), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=INK)
    for xi, (metric, _) in zip(x, metrics):
        p = summary["metricas"][metric]["p_valor"]
        ax.annotate(_format_p(p), xy=(xi, 0), xytext=(0, -20), textcoords="offset points",
                    ha="center", va="top", fontsize=6.8, color=INK_SECONDARY, annotation_clip=False)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Valor médio (%) ± desvio padrão")
    top = max(100 * (summary["metricas"][m][s]["media"] + summary["metricas"][m][s]["desvio_padrao"])
              for m, _ in metrics for s in config.SCENARIOS)
    ax.set_ylim(0, top * 1.18)
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    return _save(fig, path)
