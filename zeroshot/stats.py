import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config

METRICS = ("acuracia_top1", "f1_macro", "tempo_inferencia_ms_img", "tempo_classificador_s")


def paired_arrays(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    wide = df.pivot(index="seed", columns="cenario", values=metric).sort_index()
    return wide["H0"].to_numpy(), wide["H1"].to_numpy()


def describe(values: np.ndarray) -> dict[str, float]:
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "media": float(values.mean()),
        "desvio_padrao": float(values.std(ddof=1)),
        "mediana": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "minimo": float(values.min()),
        "maximo": float(values.max()),
    }


def paired_test(h0: np.ndarray, h1: np.ndarray, alpha: float = config.ALPHA) -> dict:
    diff = h1 - h0
    shapiro = stats.shapiro(diff)
    normal = bool(shapiro.pvalue > alpha)
    if normal:
        result = stats.ttest_rel(h1, h0)
        test_name = "t de Student pareado"
    else:
        result = stats.wilcoxon(h1, h0)
        test_name = "Wilcoxon pareado (postos sinalizados)"
    sd = diff.std(ddof=1)
    sem = sd / np.sqrt(len(diff))
    t_crit = stats.t.ppf(1 - alpha / 2, df=len(diff) - 1)
    return {
        "n_pares": int(len(diff)),
        "H0": describe(h0),
        "H1": describe(h1),
        "diferenca_H1_menos_H0": describe(diff),
        "ic95_media_diferenca": [float(diff.mean() - t_crit * sem), float(diff.mean() + t_crit * sem)],
        "rodadas_H1_maior": int((diff > 0).sum()),
        "rodadas_empate": int((diff == 0).sum()),
        "rodadas_H0_maior": int((diff < 0).sum()),
        "shapiro_wilk": {"W": float(shapiro.statistic), "p_valor": float(shapiro.pvalue), "normal": normal},
        "teste": test_name,
        "estatistica": float(result.statistic),
        "p_valor": float(result.pvalue),
        "alpha": alpha,
        "rejeita_H0": bool(result.pvalue < alpha),
        "cohen_d_pareado": float(diff.mean() / sd) if sd > 0 else float("nan"),
    }


def shift_breakdown(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        shift: {s: float(df.loc[df["cenario"] == s, f"acuracia_{shift}"].mean()) for s in config.SCENARIOS}
        for shift in config.SHIFTS
    }


def summarize(df: pd.DataFrame, setup: dict) -> dict:
    return {
        "configuracao": setup,
        "metricas": {metric: paired_test(*paired_arrays(df, metric)) for metric in METRICS},
        "acuracia_media_por_shift": shift_breakdown(df),
    }


def save_summary(summary: dict, path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
