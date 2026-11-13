from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "resultados"
FIGURES_DIR = ROOT / "figuras"

MODEL_ID = "openai/clip-vit-base-patch32"
IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

N_SIMULATIONS = 30
SAMPLES_PER_CLASS = 5
BATCH_SIZE = 128
ALPHA = 0.05
SHIFT_ASSIGNMENT_SEED = 0

SHIFTS = ("ruido_gaussiano", "desfoque_gaussiano", "cinza_baixo_contraste", "esboco")
NOISE_SIGMA = 0.08
BLUR_SIGMA = 0.7
CONTRAST_FACTOR = 0.4

SCENARIOS = ("H0", "H1")
SCENARIO_LABELS = {"H0": "H0: prompt único", "H1": "H1: prompt ensembling"}

TEMPLATES = {
    "H0": ("a photo of a {}.",),
    "H1": (
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a sketch of a {}.",
        "a low contrast photo of a {}.",
        "a cropped photo of a {}.",
        "a bright photo of a {}.",
        "a dark photo of a {}.",
        "a painting of a {}.",
    ),
}
