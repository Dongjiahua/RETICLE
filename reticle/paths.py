"""Where RETICLE reads data and writes results.

RETICLE_DATA points to the data root (benchmarks, expert checkpoints, the evidence library and run outputs); the layout below it
is described in README.md. Model weights are read from the Hugging Face cache (HF_HOME) unless a config gives a local path."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('RETICLE_DATA', REPO/'data'))
CONFIGS = REPO/'configs'
