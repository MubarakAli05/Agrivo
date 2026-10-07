# AgriMini v0

An agricultural AI **proof-of-working**, implemented in reviewed phases. **Phases 1–3 are implemented:** repository/configuration, non-destructive setup/status, a validated source/license registry, and tiny deterministic synthetic soil/plant-metadata/QA fixtures. Official source pages were inspected on 2026-10-07; no external datasets were ingested. Retrieval adapters, text tokenization, transformer training, image analysis, live question answering, and the dashboard are not implemented yet. There are no trained weights or model evaluation metrics.

The existing soil tokenizer remains unchanged and usable independently (see below).

## Phase 1 setup

Use Python 3.10 or newer (validated on Python 3.12). Run from this repository root. Phase 2 adds PyYAML for safe YAML parsing; the existing soil tokenizer is still dependency-free:

```powershell
python -m pip install -e .
python -m agri.cli setup
python -m agri.cli status
python -m agri.cli inspect-sources
python -m agri.cli inspect-sources --source soilgrids
python -m unittest -v test_soil_tokenizer tests.test_phase1 tests.test_source_registry tests.test_synthetic_data
```

Commands print JSON. RED (invalid/missing configuration or registry) returns exit code 1; GREEN and YELLOW return 0. YELLOW means metadata inspection succeeded but usage approval remains blocked, not that ingestion is permitted. Setup creates missing directories, [configuration](configs/agri-mini.json), and the source/license registry pair; it preserves existing files and rejects invalid content rather than replacing it. Status and inspection are read-only. For a separate data workspace, add `--root C:\path\to\workspace` to any command. The default root is the source checkout containing the `agri` package; these commands are intended to run from a source checkout.

The directory scaffold includes application/API/dashboard, raw/processed/indexed/cached/versioned/quarantined data, source adapters, tokenizer, transformer/soil/vision models, retrieval, QA, training, evaluation, checkpoints, licenses, tests, and documentation. Empty directories have `.gitkeep` markers; they do **not** represent implemented components. Generated datasets and checkpoints are excluded by [.gitignore](.gitignore).

### Configuration and resource policy

[Configuration validation](agri/config.py) rejects unknown or missing keys, malformed types, incompatible attention dimensions, and pretrained-model settings. Defaults are offline, CPU, seed 42, and a planned randomly initialized 4-layer/256-hidden/4-head transformer with vocabulary 2,048 and context 256. Training defaults specify a 20-step, one-epoch, batch-size-two dry run with checkpoints every ten steps. These are **future trainer settings**, not measured results or an implemented training/resource guard.

Setup never accesses the network, installs PyTorch, creates credentials, downloads datasets/checkpoints, or starts training. Later training phases must first report dataset size, storage, GPU memory, expected duration, checkpoint frequency, epochs, and data sufficiency, then pass a tiny dry run before any approved larger run. Source ingestion must wait for source inspection and license approval. Keep credentials out of configuration and version control.

Available commands are `setup`, `status`, `inspect-sources`, `generate-synthetic`, and `validate-data`. Later commands are deliberately unavailable rather than returning simulated results. Status reports the Phase 1 layout gate, Phase 2 registry/review gate, and (when a release exists) Phase 3 fixture gate separately. Overall status remains YELLOW while external approvals are pending, even when Phase 3 is GREEN; none proves end-to-end model success.

## Phase 2: source registry

[Source registry](data/source_registry.yaml) and [license registry](licenses/registry.json) contain all ten requested sources, original URLs, configurable priorities (1 highest), access modes with evidence and verification state, inspection dates, source-version missingness, declared licenses, and intended-use approvals. They are bootstrapped from [the inspected catalog](agri/source_catalog.py) without overwriting existing files. Unknown data versions remain null: a documentation date, repository branch, or `latest` URL is not a pinned dataset release.

`inspect-sources` reads the **stored inspection**, makes no HTTP requests, and reports SHA-256 fingerprints of the two local registry files. These hashes are **not dataset checksums**, and documented access modes do not establish current endpoint health. Raw-file checksums, retrieval timestamps, sizes/counts, transformation history, adapter implementations, and cache/snapshot fallback belong to later ingestion phases. Re-running inspection does not refresh the manual inspection date.

| Source | Observed evidence | Ingestion |
|---|---|---|
| SoilGrids | ISRIC declares CC-BY-4.0; official docs report REST paused and document WCS/WebDAV alternatives | Disabled; intended-use review required |
| ISRIC catalog | Multiple resources; no blanket dataset license established | Blocked; select a specific dataset |
| SSURGO | Portal documents spatial/tabular downloads; application permissions are not dataset terms | Blocked; dataset terms unresolved |
| data.gov.in | Landing and license pages returned HTTP 403 during inspection | Blocked; access and terms unresolved |
| PlantVillage | Repository dataset card declares CC-BY-SA-3.0 and documents leaf grouping | Disabled; distribution scope/intended-use review required |
| PlantDoc | README/license declare CC-BY-4.0; images described as internet-scraped | Disabled; upstream rights/intended-use review required |
| Plant Pathology 2020/2021 | Landing and rules pages exposed only titles to inspection | Blocked; competition terms unresolved |
| AI Challenger agriculture | Supplied URL returned HTTP 404 | Blocked; no substitute assumed |
| Kaggle discovery | Search page, not a single licensed dataset | Blocked; register specific datasets separately |

License declarations are evidence, **not legal clearance**. Commercial/redistribution terms are recorded only where the inspected license explicitly states them; unknown rights are never inferred from public availability. All sources start disabled and all approvals pending. Three license declarations are recorded; seven remain unresolved. Each source's unresolved-license gate is RED even though registry inspection itself succeeds with YELLOW.

[Registry validation and eligibility checks](agri/source_registry.py) reject missing/mismatched licenses, duplicate YAML/JSON keys, unsafe YAML tags, unsupported schema versions, invalid URLs, and incomplete approvals. Dates in YAML must be quoted `YYYY-MM-DD` strings. A future ingestion adapter must call `require_ingestion_approval(root, source_id, purpose)` before fetching data: it requires an enabled non-catalog source, a reviewed source, a documented acquisition mode, declared dataset license terms, and a named/dated approval matching `research_training`, `commercial_training`, or `redistribution`. Commercial use/redistribution also require explicitly established corresponding rights. Enabling a source alone never approves it. No approval is granted automatically, and the gate does not itself implement downloading or check live availability.

External ingestion remains blocked independently of synthetic development.

## Phase 3: tiny synthetic fixtures

```powershell
python -m agri.cli generate-synthetic
python -m agri.cli validate-data
python -m agri.cli status
```

[The generator](agri/synthetic_data.py) creates an immutable release at `data/releases/synthetic-v0.1-seed42/` using the configured seed. Setup does not generate it automatically. Generation requires valid configuration and external registry files, but it does not approve, modify, or ingest any external source. All work is local and CPU-only; no model, image, or network request is involved.

| Content | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Synthetic soil records | 16 | 4 | 4 | 24 |
| Fictional plant metadata | 4 | 1 | 1 | 6 |
| Fixture-derived QA examples | 72 | 18 | 18 | 108 |

The 12 fictional soil locations have two depth intervals each (0–5 and 5–15 cm). Whole location and plant-metadata groups stay in one split, and every QA context/citation resolves within that split. Repeated QA templates are intentional: these are integration fixtures, **not an independent generalization benchmark**. There are no images, image checksums, real plant identities, real disease labels, real coordinates, or observation timestamps.

[The soil schema](agri/data_schema.py) defines 12 numeric fields with explicit units, nulls and boolean missingness masks. Valid zero values remain present. Phosphorus is deliberately all missing; pH, potassium, moisture, and EC also contain missing examples. Texture fractions total 100%. Any pH uncertainty is a **fabricated, uncalibrated test interval**, never measured uncertainty. These values must not be used for farming recommendations.

Release contents:

```text
soil.jsonl
plants.jsonl
schema.json
manifest.json
qa/train.jsonl
qa/validation.jsonl
qa/test.jsonl
```

The manifest records source `agrimini_synthetic_fixture`, dataset/generator versions, seed, UTC generation time, counts, splits, per-file SHA-256 checksums/sizes, and transformation history. Original URL, download timestamp, and external license are null because no external dataset was used; no commercial license is inferred. This local manifest is the synthetic provenance record, separate from the ten-source external registry.

QA covers values, units, missingness, uncertainty, provenance, comparisons, schema definitions, dataset labels, and unsupported location/image questions. **34 examples expect UNKNOWN.** `synthetic_verified` means consistent with these fixtures—not externally verified agricultural facts, model accuracy, or a passed hallucination evaluation.

`validate-data` checks the fixed file inventory, deterministic payload bytes, manifest, schema, group isolation, and QA evidence. Missing, extra, altered, or corrupt files fail with RED; generation never silently replaces an existing release. Re-running generation preserves a valid release, including its timestamp. Changing the configured seed creates a separate release. Data bytes reproduce for the same seed/generator; the initial manifest generation time differs between independently created releases. Generated releases are ignored by Git and can be recreated from the tracked generator; raw source files are never modified.

[Fixture tests](tests/test_synthetic_data.py) cover determinism, global-RNG isolation, provenance, group splits, missingness/zeros, invalid schemas, UNKNOWN answers, immutable publication, corruption detection, CLI behavior, and read-only status. The fixture is intentionally below 1 MB and is not enough data to establish model quality.

Next: review Phase 3, then Phase 4 (custom tokenizer). Fit vocabulary/numeric bins on **training records only**. The existing numeric tokenizer rejects all-missing training features; Phase 4 must explicitly handle the deliberate phosphorus case rather than invent values or fit on held-out records.

## Existing soil-health numeric tokenizer

A dependency-free Python tokenizer for structured soil measurements. It learns numeric bins from training data and produces fixed-length lists of integer token IDs for embedding-based ML models. It does not predict soil health or define agronomic safety thresholds.

## Usage

Run with Python 3.10 or newer (validated on Python 3.12).

```python
from soil_tokenizer import SoilTokenizer

# Illustrative data only. Use your actual training split for fitting.
training_rows = [
    {"ph": 5.5, "nitrogen": 20, "phosphorus": 10, "potassium": 100},
    {"ph": 6.5, "nitrogen": 40, "phosphorus": 20, "potassium": 150},
    {"ph": 7.5, "nitrogen": 60, "phosphorus": 30, "potassium": 200},
]

tokenizer = SoilTokenizer(n_bins=4)
train_ids = tokenizer.fit_transform(training_rows)
validation_ids = tokenizer.transform([
    {"ph": 6.2, "nitrogen": 35, "phosphorus": None, "potassium": 130}
])
assert validation_ids == [[3, 8, 11, 18]]

# Use vocab_size as the embedding table size; each row has four tokens.
print(tokenizer.vocab_size)  # 21
print(validation_ids)

tokenizer.save("soil_tokenizer.json")
restored = SoilTokenizer.load("soil_tokenizer.json")
assert restored.transform(training_rows) == train_ids
```

CSV input works directly with the standard library:

```python
import csv
from soil_tokenizer import SoilTokenizer

with open("training.csv", newline="", encoding="utf-8") as stream:
    tokenizer = SoilTokenizer()
    train_ids = tokenizer.fit_transform(csv.DictReader(stream))
```

## Data contract

- Default feature order: `ph`, `nitrogen`, `phosphorus`, `potassium`. Names are case-sensitive. Customize with `SoilTokenizer(features=["ph", "organic_matter", "moisture"], n_bins=16)`.
- Each input row is a mapping from feature names to numeric values or numeric strings. Unconfigured keys (such as labels and sample IDs) are ignored.
- Use consistent units, measurement methods, and column meanings across training and inference. The tokenizer does not convert units or validate physical ranges. Soil pH is unitless; nutrient units must be chosen to match your dataset.
- Missing keys, `None`, blank strings, and NaN become feature-specific missing tokens. Booleans, malformed values, and infinity raise `ValueError`.
- Every configured feature must have at least one nonmissing training value. Empty training input fails.
- Split your dataset **before** fitting. Call `fit` only on training data, then `transform` on validation, test, and production data. Save the tokenizer alongside the model; refitting changes token meanings.

## Bins and token IDs

Bins use empirical nearest-rank quantiles. Values equal to a boundary belong to the lower bin. Duplicate boundaries collapse, and constant features have one numeric bin. There can be fewer effective bins than `n_bins`, especially with small or tied datasets. Values outside the training range map to the first or last effective bin without changing boundaries.

Token `0` is reserved for padding and is never emitted. For zero-based feature index `i`, the missing token is `1 + i * (n_bins + 1)`; numeric bin `b` uses `2 + i * (n_bins + 1) + b`. Each feature has a disjoint ID range. `vocab_size` is `1 + number_of_features * (n_bins + 1)`, including reserved but unused bins.

Missing measurements are actual tokens, not padding. For a transformer, an attention mask for these fixed-length sequences can contain all ones. Convert the returned lists to your framework's integer tensor type (for example, PyTorch `torch.long`) before embedding. Quantization is lossy; retain continuous measurements separately if your model needs exact values. Tree-based models often work better with continuous numeric columns rather than token IDs.

## Validation

```powershell
python -m unittest -v test_soil_tokenizer
```

Tests cover boundaries, feature order, missing values, malformed inputs, constant and tied data, out-of-range inference, train/inference separation, failed refits, and JSON persistence.
