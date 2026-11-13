import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from . import config
from .data import ShiftedDataset, build_shifted_batch, stratified_sample, to_clip_input
from .model import ZeroShotCLIP


def _scenario_order(seed: int) -> tuple[str, ...]:
    return config.SCENARIOS if seed % 2 else config.SCENARIOS[::-1]


def run_round(clip: ZeroShotCLIP, data: ShiftedDataset, seed: int) -> list[dict]:
    indices = stratified_sample(data.labels, config.SAMPLES_PER_CLASS, seed)
    pixel_values = to_clip_input(build_shifted_batch(data, indices))
    y_true = data.labels[indices]
    shifts = data.shifts[indices]
    labels = np.arange(len(data.classnames))
    rows = []
    for scenario in _scenario_order(seed):
        classifier, text_time = clip.timed_classifier(data.classnames, config.TEMPLATES[scenario])
        predictions, image_time = clip.timed_predict(pixel_values, classifier)
        y_pred = predictions.numpy()
        row = {
            "seed": seed,
            "cenario": scenario,
            "n_imagens": len(indices),
            "n_templates": len(config.TEMPLATES[scenario]),
            "acuracia_top1": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "tempo_inferencia_ms_img": 1000.0 * image_time / len(indices),
            "tempo_classificador_s": text_time,
        }
        for k, shift in enumerate(config.SHIFTS):
            mask = shifts == k
            row[f"acuracia_{shift}"] = accuracy_score(y_true[mask], y_pred[mask])
        rows.append(row)
    return rows


def run_simulations(clip: ZeroShotCLIP, data: ShiftedDataset, n: int = config.N_SIMULATIONS) -> pd.DataFrame:
    run_round(clip, data, seed=0)
    rows = [row for seed in tqdm(range(1, n + 1), desc="Simulações") for row in run_round(clip, data, seed)]
    return pd.DataFrame(rows).sort_values(["seed", "cenario"]).reset_index(drop=True)
