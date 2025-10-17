# deepparse-artifact

> **Acceptance Criteria**
> 1. `make setup && ./scripts/run_demo.sh` completes on a blank CPU machine in < 3 minutes and prints GA/PA for a toy dataset.
> 2. `python -m deepparse.cli synth --dataset <name> --k 50 --mode offline` creates `artifacts/masks/<name>.json` with ≥4 canonical regex classes (timestamp, IP, numeric, level) and passes regex validation.
> 3. `python -m deepparse.cli eval --config configs/eval_16_datasets.yaml` produces per-dataset GA/PA CSVs and a macro-average row; DeepParse columns are populated.
> 4. `python -m deepparse.cli time --dataset <name> --n 100` outputs a timing CSV suitable for Table II.
> 5. `./scripts/regenerate_tables.sh` emits `table_I.tex` and `table_II.tex` whose numbers match the produced CSVs.
> 6. All tests in `tests/` pass on CI.

## Abstract
DeepParse is a hybrid log parsing system that combines one-time large language model (LLM) assisted synthesis of regex masks with deterministic Drain parsing. This repository packages the full artifact for the paper “DeepParse: A Hybrid LLM-Enhanced Framework for Accurate Log Parsing,” providing scripts, configuration files, datasets layout, evaluation harnesses, and reproducible outputs for reviewers. All computations are deterministic, seeds are logged for every run, and the codebase is double-blind ready (no names or telemetry).

## Architecture Overview
```
+---------------------+          +--------------------+
| k-sample selection  | --50-->  | LLM/stub mask gen  |
+---------------------+          +--------------------+
            |                              |
            v                              v
    artifacts/masks/*.json      validated regex masks
            |                              |
            +--------------v---------------+
                           |
                   Drain parser engine
                           |
                    parsed templates
                           |
             +----------------------------+
             | Metrics (GA/PA) & timing   |
             +----------------------------+
```

## Repository Layout
The tree below highlights the most relevant folders. All generated artifacts (logs, CSVs, LaTeX tables) live under `artifacts/outputs/` and are gitignored.

- `deepparse/`: Source code for the CLI, mask synthesis, Drain parser, metrics, evaluation harness, utilities, and deterministic helpers.
- `configs/`: YAML configuration snapshots for demo and 16-dataset evaluation.
- `scripts/`: Shell helpers for environment setup, dataset fetching, evaluation orchestration, and table regeneration.
- `artifacts/`: Placeholder directories for datasets, outputs, and synthesized masks.
- `tests/`: Unit and integration tests with a golden snapshot for CLI output.
- `anonymization/`: Utilities to scrub host-specific paths in generated artifacts.

## Getting Started
### Prerequisites
- Python 3.11
- Git
- Optional: Conda, Docker, GNU Make, UV package manager.

### Environment Setup Options
We provide three reproducible setup pathways:

1. **Makefile/UV workflow (recommended):**
   ```bash
   make setup
   ```
2. **pip-tools/UV lock install:**
   ```bash
   uv sync
   ```
3. **Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate deepparse-artifact
   ```
4. **Docker:**
   ```bash
   docker build -t deepparse-artifact .
   docker run --rm -it -v $PWD:/workspace deepparse-artifact
   ```

All pathways ensure deterministic dependencies pinned to versions tested in CI.

## Reproduction Tiers
DeepParse supports three reproducibility tiers to accommodate different resource levels.

### Tier A – No GPU Demo (≤3 minutes)
```bash
make setup
./scripts/prepare_paths.sh
./scripts/run_demo.sh
```
This mode uses the offline regex stub and a tiny bundled dataset to demonstrate mask synthesis, parsing, and GA/PA metrics entirely on CPU without internet access.

### Tier B – CPU-only Full Evaluation
```bash
./scripts/prepare_paths.sh
./scripts/fetch_datasets.sh        # or manually place data
python -m deepparse.cli synth --dataset ALL --k 50 --mode offline
python -m deepparse.cli eval --config configs/eval_16_datasets.yaml --deterministic
python -m deepparse.cli table --inputs artifacts/outputs/*.csv --out artifacts/outputs/tables/
./scripts/regenerate_tables.sh
```
Runs Drain parsing over all 16 corrected LogHub datasets, computing GA/PA metrics and reproducing Table I/II numbers. Expect longer runtimes on CPU but identical outputs to GPU mode.

### Tier C – GPU Fast Path
```bash
python -m deepparse.cli eval --config configs/eval_16_datasets.yaml --deterministic --device cuda
```
When CUDA is available, Drain’s tensorized similarity checks and optional HF mode mask refinement leverage the GPU for faster processing. Determinism is preserved by enabling `torch.backends.cudnn.deterministic` when `--deterministic` is passed.

## Command Line Interface
The CLI bundles four subcommands:

- `synth`: Generate regex mask lists using the offline stub or optional Hugging Face pipeline.
- `parse`: Apply masks and Drain parser to produce structured templates for a dataset.
- `eval`: Run the entire benchmark, computing GA and PA metrics for each dataset and macro averages.
- `time`: Benchmark parsing throughput on 100 logs (Table II).
- `table`: Convert CSV outputs into LaTeX tables.

See `python -m deepparse.cli --help` for the full argument list.

## Programmatic API
For quick experiments the package exposes a minimal API aligned with the paper’s
Figure 2 workflow:

```python
from deepparse import Drain, synth_masks

patterns = synth_masks(sys_logs, sample_size=500, temperature=0, num_beams=2, max_length=512)
drain = Drain()
drain.load_masks(patterns)
parsed_templates = drain.parse_all(sys_logs)
```

`synth_masks` returns a list of dictionaries containing `label`, `pattern`, and
`justification` fields. The offline mode is fully deterministic; switching to
`mode="hf"` enables the optional Hugging Face pipeline with the same control
parameters (`temperature`, `num_beams`, `max_length`).

## Data Availability
The corrected LogHub datasets are public but must be downloaded separately to preserve double-blind review. Use `./scripts/fetch_datasets.sh` for online retrieval or copy the datasets into `artifacts/data/<DATASET_NAME>/` manually. Checksums are verified on load; mismatches trigger actionable error messages.

## Determinism & Seeds
Deterministic behavior is enforced across Python’s `random`, NumPy, and PyTorch (if available). The `deepparse.seeds.set_global_seed` function applies consistent seeds, logs the seed value, and optionally sets deterministic CUDA flags. CLI commands expose a `--seed` option for reproducibility and record seeds in `artifacts/outputs/logs/`. Sampling of k logs uses a stratified deterministic algorithm to ensure identical subsets across runs.

## Double-Blind Compliance Checklist
- Repository name and contents omit author identities.
- No telemetry, analytics, or network beacons are included.
- Offline workflows are fully supported.
- The anonymization scripts scrub hostnames and absolute paths before logs are stored.

## Mask Regeneration Guidance
When log schemas drift, regenerate masks with:
```bash
python -m deepparse.cli synth --dataset <name> --k 50 --mode offline --out artifacts/masks/<name>.json
```
Validate using the built-in regex compilation checks (`--strict` to forbid greedy `.*`). For best results, refresh the mask whenever new log templates are introduced.

## Troubleshooting
- **Regex validation failure:** Check `artifacts/outputs/logs/*.log` for the offending pattern and adjust the mask generator prompt or stub heuristics.
- **Slow I/O:** Use `--workers` on the CLI to parallelize dataset loading; the CPU pipeline is I/O bound for large datasets.
- **Missing CUDA:** Run with `--device cpu`; the deterministic flag forces CPU execution when GPUs are absent.
- **Dataset missing:** Ensure the dataset folder under `artifacts/data/` matches the dataset name in configs.

## Open Science Assets
All CSV results, LaTeX tables, and logs are written to `artifacts/outputs/`. Config snapshots used to reproduce experiments are kept under `configs/`. The repository includes `THIRD_PARTY_NOTICES.md` listing all third-party dependencies and licenses.

## Regenerating Figures and Tables
Run:
```bash
./scripts/regenerate_tables.sh
```
This command rebuilds Table I and Table II LaTeX/CSV files from the most recent evaluation outputs.

## Continuous Integration
GitHub Actions (`.github/workflows/ci.yml`) run linting (`ruff`), unit tests (`pytest -q`), and the tiny demo pipeline to guarantee reproducibility and correctness for every commit.

## Citation
See `CITATION.cff` for how to cite this artifact.

## License
Licensed under the Apache License, Version 2.0. See `LICENSE` for details.
