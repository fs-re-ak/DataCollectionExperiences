"""
Global configuration constants for the Emotions Model Experience.
All timing values are in seconds unless otherwise noted.
"""

# ── Device ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 250          # Hz  (ADS1299 via Hermes V1)
N_CHANNELS  = 8            # 7 EMG + 1 temporal EEG

CHANNEL_LABELS = [
    "Supraorbital_L",   # ch1 – EMG  – Anger/Fear/Sadness
    "Supraorbital_R",   # ch2 – EMG  – Anger/Fear/Sadness
    "Zygomatic_L",      # ch3 – EMG  – Happiness/Contempt
    "Zygomatic_R",      # ch4 – EMG  – Happiness
    "Temporal_EEG",     # ch5 – EEG  – reference temporal
    "Glabella",         # ch6 – EMG  – Disgust/Anger
    "Temporal_R",       # ch7 – EMG  – Contempt
    "Nasolabial",       # ch8 – EMG  – Disgust/Surprise
]

# ── Trial timing ──────────────────────────────────────────────────────────────
T_FIXATION = 2.0    # instruction / setup phase
T_HOLD     = 6.0    # emotion expression hold
T_RELAX    = 3.0    # relax / return to neutral
T_TRIAL    = T_FIXATION + T_HOLD + T_RELAX  # 11 s total

# ── Session structure ─────────────────────────────────────────────────────────
T_BASELINE_PRE  = 30.0   # pre-session resting baseline
T_BASELINE_POST = 30.0   # post-session resting baseline
T_REST          = 25.0   # inter-block rest

N_BLOCKS = 3
# Which blocks use the webcam mirror (1-indexed)
MIRROR_BLOCKS = {1, 3}

# ── Emotions ──────────────────────────────────────────────────────────────────
# Base set used for randomisation (8 trials per block).
# Contempt is represented as two variants; the randomiser picks exactly one
# per block and annotates it with the instructed side.
EMOTIONS_BASE = [
    "NEUTRAL",
    "HAPPINESS",
    "ANGER",
    "DISGUST",
    "FEAR",
    "SURPRISE",
    "SADNESS",
    "CONTEMPT_LEFT",
    "CONTEMPT_RIGHT",
]

# ── French display labels ──────────────────────────────────────────────────────
EMOTION_LABELS_FR = {
    "NEUTRAL":        "NEUTRE",
    "HAPPINESS":      "JOIE",
    "ANGER":          "COLÈRE",
    "DISGUST":        "DÉGOÛT",
    "FEAR":           "PEUR",
    "SURPRISE":       "SURPRISE",
    "SADNESS":        "TRISTESSE",
    "CONTEMPT_LEFT":  "SOURIRE EN COIN (gauche)",
    "CONTEMPT_RIGHT": "SOURIRE EN COIN (droite)",
}

# ── English display labels ─────────────────────────────────────────────────────
EMOTION_LABELS_EN = {
    "NEUTRAL":        "NEUTRAL",
    "HAPPINESS":      "HAPPINESS",
    "ANGER":          "ANGER",
    "DISGUST":        "DISGUST",
    "FEAR":           "FEAR",
    "SURPRISE":       "SURPRISE",
    "SADNESS":        "SADNESS",
    "CONTEMPT_LEFT":  "SMIRK (left)",
    "CONTEMPT_RIGHT": "SMIRK (right)",
}

# ── French FACS instructions ───────────────────────────────────────────────────
EMOTION_INSTRUCTIONS_FR = {
    "NEUTRAL":        "Visage détendu, expression neutre.",
    "HAPPINESS":      "Sourire large — joues relevées, coins des lèvres tirés vers le haut et l'extérieur.",
    "ANGER":          "Sourcils froncés et abaissés, regard intense, lèvres serrées.",
    "DISGUST":        "Nez plissé, lèvre supérieure relevée, légère protrusion de la langue.",
    "FEAR":           "Sourcils relevés et rapprochés, yeux écarquillés, bouche entrouverte.",
    "SURPRISE":       "Sourcils arqués, yeux grands ouverts, bouche ouverte.",
    "SADNESS":        "Commissures des lèvres abaissées, paupières supérieures tombantes.",
    "CONTEMPT_LEFT":  "Coin gauche de la bouche relevé et tiré vers l'intérieur.",
    "CONTEMPT_RIGHT": "Coin droit de la bouche relevé et tiré vers l'intérieur.",
}

# ── English FACS instructions ──────────────────────────────────────────────────
EMOTION_INSTRUCTIONS_EN = {
    "NEUTRAL":        "Relaxed face, neutral expression.",
    "HAPPINESS":      "Broad smile — cheeks raised, lip corners pulled up and outward.",
    "ANGER":          "Brows pulled down and together, intense gaze, lips pressed tight.",
    "DISGUST":        "Nose wrinkled, upper lip raised, slight tongue protrusion.",
    "FEAR":           "Brows raised and pulled together, eyes wide open, mouth slightly open.",
    "SURPRISE":       "Brows arched high, eyes wide open, mouth open.",
    "SADNESS":        "Lip corners pulled down, upper eyelids drooping.",
    "CONTEMPT_LEFT":  "Left corner of the mouth raised and pulled inward.",
    "CONTEMPT_RIGHT": "Right corner of the mouth raised and pulled inward.",
}

# ── Networking ────────────────────────────────────────────────────────────────
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765

# ── Output ────────────────────────────────────────────────────────────────────
DATA_DIR = "data"   # base directory; each session gets its own subfolder inside
