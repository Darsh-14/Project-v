# CHANGELOG

Deviations, fallbacks and decisions, newest first.

## 2026-10-24 (Day 1)

- **Local Python is 3.13** (only interpreter installed). CLAUDE.md §3 asks for 3.10/3.11 to match
  Kaggle. Local venv is for development and tests only; all training runs on Kaggle with the image's
  Python. `requirements.lock` pins only project packages, not torch/numpy, so it installs on the
  Kaggle image.
