# AgriMini v0

An agricultural AI **proof-of-working**, implemented in independently verified phases. **Phases 1–5, 8 and 10–11 are published:** repository/configuration, source/license registry, synthetic fixtures, custom tokenizers, a manually implemented transformer, and provenance-aware local retrieval. Remaining phases are being implemented in parallel and published separately after validation. Official source pages were inspected on 2026-10-07; no external datasets were ingested. Local retrieval is not neural answering or proof of agricultural accuracy.

The existing soil tokenizer remains unchanged and usable independently (see below).

## Phase 1 setup

Use Python 3.10 or newer (validated on Python 3.12). Run from this repository root. Phase 2 adds PyYAML for safe YAML parsing; the existing soil tokenizer is still dependency-free:

```powershell
python -m pip install -e .
python -m agri.cli setup
python -m agri.cli status
python -m agri.cli inspect-sources
python -m agri.cli inspect-sources --source soilgrids
python -m unittest discover -v
```

Commands print JSON. RED (invalid/missing configuration or registry) returns exit code 1; GREEN and YELLOW return 0. YELLOW means metadata inspection succeeded but usage approval remains blocked, not that ingestion is permitted. Setup creates missing directories, [configuration](configs/agri-mini.json), and the source/license registry pair; it preserves existing files and rejects invalid content rather than replacing it. Status and inspection are read-only. For a separate data workspace, add `--root C:\path\to\workspace` to any command. The default root is the source checkout containing the `agri` package; these commands are intended to run from a source checkout.

The directory scaffold includes application/API/dashboard, raw/processed/indexed/cached/versioned/quarantined data, source adapters, tokenizer, transformer/soil/vision models, retrieval, QA, training, evaluation, checkpoints, licenses, tests, and documentation. Empty directories have `.gitkeep` markers; they do **not** represent implemented components. Generated datasets and checkpoints are excluded by [.gitignore](.gitignore).

### Configuration and resource policy

[Configuration validation](agri/config.py) rejects unknown or missing keys, malformed types, incompatible attention dimensions, and pretrained-model settings. Defaults are offline, CPU, seed 42, and a randomly initialized 4-layer/256-hidden/4-head transformer. The tokenizer vocabulary budget is 2,048; the model uses the actual fitted vocabulary. Phase 5 raises the context from 256 to **1,536**, as explicitly selected, to preserve each current QA example plus structured evidence intact. Training defaults specify a 20-step, one-epoch, batch-size-two dry run with checkpoints every ten steps. These are **future trainer settings**, not measured results or an implemented training/resource guard.

Setup never accesses the network, installs PyTorch, creates credentials, downloads datasets/checkpoints, or starts training. Later training phases must first report dataset size, storage, GPU memory, expected duration, checkpoint frequency, epochs, and data sufficiency, then pass a tiny dry run before any approved larger run. Source ingestion must wait for source inspection and license approval. Keep credentials out of configuration and version control.

Available commands are `setup`, `status`, `inspect-sources`, `generate-synthetic`, `validate-data`, `train-tokenizer`, `validate-tokenizer`, `check-model`, `validate-model`, `build-index`, `validate-index`, and `search`. Additional commands are registered only when their implementations are integrated. Status reports layout, external-source review, fixture validation, tokenizer validation, and recorded model checks separately when their artifacts exist. Overall status remains YELLOW while external approvals are pending, even when Phases 3–5 are GREEN; none proves end-to-end model success.

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

Vocabulary, numeric boundaries, and categories must be fitted on **training records only**. Phase 4 handles all-missing phosphorus explicitly, without inventing observations or using held-out values.

## Phase 4: custom text and structured tokenizers

```powershell
python -m agri.cli train-tokenizer --dry-run
python -m agri.cli train-tokenizer
python -m agri.cli validate-tokenizer
python -m agri.cli status
```

Run setup and synthetic generation first on a fresh checkout. Training is offline, CPU-only, and dependency-free beyond existing workspace dependencies. The dry run uses four training QA examples, four training soil records, and at most 320 text tokens, writes nothing, and checks roundtrips. A normal training run always passes this preflight before fitting the 72 training QA examples and 16 training soil records. Validation/test records are used only for roundtrip/encoding checks, never learning. No neural model, optimizer, epochs, GPU, downloaded vocabulary, or pretrained weights are involved.

### Text BPE

[The custom byte BPE](tokenizer/tokenizer.py) normalizes CRLF/CR to LF and Unicode to NFC, retaining case and other whitespace. It starts with all 256 UTF-8 bytes plus explicit special tokens. Weighted adjacent-pair counts learn ranked merges with minimum frequency two and deterministic ID-pair tie breaks. Merges never cross documents or regex segments (letters, digits, punctuation, underscores, or whitespace). The configured vocabulary budget is an upper bound, not a promise to invent unsupported merges; Phase 4 accepts budgets from 267 to 8,192, default 2,048.

Special IDs 0–10, in order: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<sep>`, `<mask>`, `<question>`, `<context>`, `<answer>`, `<unknown>`, `<source>`. Literal spellings inside user text are encoded as bytes—not interpreted as control IDs. [QA framing](tokenizer/train_tokenizer.py) inserts explicit BOS, question/context/answer delimiters, and EOS. Full byte coverage avoids unknown tokens for valid Unicode without learning from held-out text.

```python
from pathlib import Path
from tokenizer.tokenizer import BPETokenizer

path = Path("tokenizer") / "releases" / "agri-tokenizer-v1-seed42-vocab2048"
text = BPETokenizer.load(path)
question = "What is pH at 0–5 cm? 🌱"
ids = text.encode(question)
assert text.decode(ids) == text.normalize(question)
```

Roundtrips reproduce **normalized** text, not necessarily its original Unicode/newline representation. Lone surrogates are rejected. Decoding arbitrary IDs that form invalid UTF-8 raises `UnicodeDecodeError`; a future neural generation layer must handle incomplete byte sequences explicitly. Special tokens are skipped by default during decoding, or rendered literally with `skip_special_tokens=False`.

### Structured soil encoding

[StructuredSoilTokenizer](tokenizer/structured.py) is a new, separate implementation; the original [SoilTokenizer](soil_tokenizer.py) and its behavior remain unchanged. It learns feature-specific nearest-rank quantile bins and category vocabularies from rows explicitly marked `train`. Each record yields **41 ordered IDs**: numeric/unit/uncertainty-presence tokens for 12 fields, plus crop, region, source, source-version, and exact depth-range categories. The text and structured ID spaces are separate; Phase 5 uses separate embedding tables, never interchanging their IDs.

Every output also retains the exact raw value, unit, missingness, uncertainty interval/method, depth, and category/source values as side channels. Bin IDs alone are lossy and cannot reconstruct measurements; Phase 5 consumes numeric side channels through feature-specific projections while retaining raw JSON text and metadata. Values outside the observed training range are flagged and retained, not clipped. The numeric `calibrated` flag means training observations exist for binning, not scientific or agronomic calibration. Units must match the canonical schema; no implicit conversions occur. Unknown categories and explicitly missing categories have distinct tokens.

**All-missing phosphorus policy:** null retains its missing token. A future nonmissing value—including zero—is preserved and receives a distinct **uncalibrated** token, with no invented quantile boundaries or calibration claim. This is the selected policy; such inputs do not refit the tokenizer. The same rule applies to any all-missing feature.

### Artifacts, validation, and limits

[The training workflow](tokenizer/train_tokenizer.py) atomically publishes an immutable release under `tokenizer/releases/agri-tokenizer-v1-seed42-vocab2048/`:

```text
vocab.json       # explicit special IDs and hexadecimal byte-sequence IDs
merges.txt       # version header and ranked ID pairs
config.json      # BPE format and normalization settings
structured.json # train-only quantiles, observed ranges, categories, units
manifest.json   # training-input hashes, settings, source, timestamp, file hashes/sizes
```

Changing seed or vocabulary budget selects a separate release. Repeating training reuses a valid release without modifying bytes or timestamps. Corruption, missing/extra files, malformed artifacts, and provenance mismatches fail rather than being repaired. Status never trains or regenerates. Generated tokenizer files are Git-ignored; checked-in code and tests reproduce them. Byte outputs are deterministic for the same inputs/runtime; only independently created manifest timestamps differ. Unicode normalization/segmentation follows the Python runtime's Unicode database.

Validation reports roundtrip failures, unknown IDs, sequence lengths, and examples exceeding the configured model context, per split. **Nothing is silently truncated.** Passing these checks establishes tokenizer mechanics—not agronomic validity, model accuracy, or hallucination resistance.

The validated seed-42 fixture release on Python 3.12 learns **760 text tokens** within the 2,048-token budget and **272 structured tokens**. All 108 QA examples roundtrip; their fully framed text sequences are 263–1,325 tokens. All exceeded the original Phase 4 context of 256. Phase 5 resolves this with the selected 1,536-position context, actual text-vocabulary output size, and separate structured embeddings; no tokenizer refit is needed.

Tests cover byte BPE, deterministic merges, Unicode, special-token isolation, strict artifact loading, structured ranges/missingness/uncertainty, train-only fitting, dry runs, atomic publication, checksums, and CLI/status behavior.

## Phase 5: custom transformer and intact context packing

Install the optional model dependency for model execution and the full test suite. Earlier workspace/tokenizer commands and read-only model validation do not import PyTorch:

```powershell
python -m pip install -e ".[model]"
python -m agri.cli check-model --dry-run
python -m agri.cli check-model
python -m agri.cli validate-model
python -m agri.cli status
```

On a fresh checkout, run setup, synthetic generation, and tokenizer training first. Review the preflight resource estimates before the configured-size check. The dry run checks all QA packing and runs a constructed 16-position, 1-layer/16-hidden forward/backward probe; it writes nothing. The normal check repeats that preflight, checks short-sequence gradients in the configured architecture, and runs one full longest **training** example through the model without gradients. There are **zero optimizer steps**, no training epochs, no generation, no downloaded weights, and no neural quality measurements. Validation/test examples are encoded only to verify that they fit, not fed through the model or used to update anything.

### Architecture and mixed input stream

[The custom model](models/transformer/model.py) implements learned token/position embeddings, feature-specific numeric projections, pre-normalized residual blocks, manual multi-head QKV attention with causal/key-padding masks, a two-layer GELU feed-forward network, manual population-variance layer normalization, dropout, and an untied text output projection. It uses PyTorch primitives, not `nn.Transformer`, `nn.MultiheadAttention`, pretrained components, or Hugging Face models.

The default architecture has **4 layers, hidden size 256, 4 heads, feed-forward size 1,024, dropout 0.1, and context 1,536**. The fitted text vocabulary is **760** and the structured vocabulary **272**. Only the 760 real text IDs are output classes; unused IDs from the 2,048-token tokenizer budget cannot be generated. The model has **4,038,144 parameters** (**16,152,576 FP32 parameter bytes**), excluding gradients and intermediate tensors.

[Context packing](models/transformer/packing.py) preserves the entire normalized question, full JSON evidence, answer, raw structured metadata, and source/evidence identifiers. The stream is:

```text
<BOS> <QUESTION> question <CONTEXT>
    [<SOURCE> 41 structured positions <SEP>] for each soil record
    complete JSON context <ANSWER> answer <EOS>
```

Text and structured IDs occupy separate tensors/embedding tables at matching stream positions. Structured positions have text ID zero and text positions have structured ID zero. Each soil record contributes exactly 43 additional positions including boundaries. Plant examples retain their complete metadata as text; no image encoder or disease prediction is implied. Records must stay inside the example's group/split and declared source/evidence boundaries. No live retriever exists yet.

At each numeric value position, seven channels represent value, uncertainty lower/upper bounds, missingness, uncertainty missingness, uncalibrated binning, and outside-training-range status. Values and bounds use a signed `log1p` transform; no fitted scaling or clipping is applied. Depth gets its own feature projection using top/bottom bounds in the lower/upper channels. Original values, uncertainty methods, units, and depth remain in the full JSON and returned raw side channels. Tensor conversion is FP32 and is not an exact reversible representation of arbitrary numbers. Missing zero and observed zero remain distinct; observed phosphorus stays explicitly uncalibrated when no training observations exist.

`pack_example` returns aligned input IDs, structured IDs, numeric features/values, attention masks, and labels, plus evidence metadata. `collate_examples` right-pads batches from one split. Import `TransformerConfig`, `AgriTransformer`, and `causal_lm_loss` from `models.transformer.model`. Pass the collated tensors except `labels` to the model; logits have shape `[batch, positions, actual_text_vocab]`. The loss shifts aligned labels internally for next-token prediction and ignores structured/padded targets (`-100`). This is a causal language-model objective, not an implemented answer-only training policy. All-padding inputs and empty supervision remain finite. Invalid tensor shapes/dtypes/ranges, conflicting namespaces, nonfinite numeric inputs, and overlong sequences are rejected.

### Validation results and boundaries

All **108** complete examples fit the selected 1,536-position context without truncation or chunking:

| Split | Examples | Minimum positions | Maximum positions |
|---|---:|---:|---:|
| Train | 72 | 263 | 1,401 |
| Validation | 18 | 266 | 1,411 |
| Test | 18 | 266 | 1,404 |

Future oversized evidence is an explicit error, never silently discarded. Smaller contexts remain configurable, but Phase 5 validation fails if the intact examples do not fit. CPU checks are bounded to at most 4 layers, hidden 256, 8 heads, FF 1,024, context 1,536, and text budget 8,192; this is not a general training resource controller.

The default CPU check passed on Python 3.12/PyTorch **2.2.2+cpu**: finite short-sequence gradients and a full 1,401-position training-example forward. The reported check took **15.322 seconds wall / 13.188 seconds CPU**, using one thread for model execution. Native peak RAM was **not measured**; preflight budgets up to 2 GiB process RAM as an estimate, not an enforced limit. A single attention-score matrix at batch one/context 1,536 is 36 MiB per layer; training must account separately for gradients, additional matrices, activations, and optimizer state. No GPU was used. The local report is **2,917 bytes**; no model weights were written.

The existing PyTorch installation emits a NumPy 2 ABI compatibility warning. Tensor-only checks pass and do not use NumPy conversion. The shared environment was not modified; NumPy interoperability is not validated.

[The check workflow](agri/model_check.py) atomically writes a Git-ignored report under `models/transformer/reports/phase5.json`, recording configuration, code/data/tokenizer hashes, runtime, checks, and packing statistics. `validate-model` and status verify the report against current artifacts/code without initializing or fitting a model. Missing, corrupt, or stale reports fail; rerun `check-model` explicitly to replace them after changes. Checksums detect accidental changes, not authenticated attestations. The report is not a checkpoint or proof of model quality. Random initialization is seeded inside an isolated CPU RNG scope, and the caller's RNG state/thread count are restored.

**155 tests pass**, including causal independence, future numeric isolation, padding, gradients, dropout, state-dict roundtrip, numeric zero/missingness, intact text/evidence preservation, overflow rejection, read-only status, provenance checks, and the prior-phase regressions. Pylance workspace diagnostics are clean. Phase 5 mechanics are GREEN; external-source approval remains YELLOW. These are Phase 5 results, before subsequent training or retrieval work.

Independent remaining implementations run in parallel. Each verified phase is committed and pushed separately; publication order need not match phase numbering when dependencies are independent.

## Phase 8: local evidence retrieval

```powershell
python -m agri.cli build-index
python -m agri.cli validate-index
python -m agri.cli search --question "What is pH at 0-5 cm for synthetic-site-02?"
```

[The local index](retrieval/index.py) contains **30 synthetic evidence records** (24 soil, six plant metadata), partitioned into 20 train / five validation / five test. **Zero QA answers** are indexed. Retrieval defaults to train and requires exact fictional entity/property/depth matching; unsupported or ambiguous questions return no evidence rather than fabricated matches. Results preserve record/source/manifest checksums, original values, units, depth, uncertainty, missingness and explicit synthetic provenance.

The actual index was built and read-only validation passed. The example retrieves synthetic pH **6.98**, not a measured field value. Index mechanics are GREEN; real-data access and agricultural quality remain blocked. No model training, network or GPU is involved. CPU time/peak RAM were not measured. The ignored index is `data/indexed/synthetic-v1.json`; rerun `build-index` explicitly after valid data changes. Next: checkpoint-backed integration and separate external-source approval.

## Phase 10: gated plant-metadata snapshot adapters

```powershell
python -m agri.cli inspect-adapter --source plantvillage
python -m agri.cli adapter-schema --source plantdoc
# This remains blocked until a named, dated, purpose-specific review is recorded:
python -m agri.cli ingest-snapshot --source plantvillage --snapshot C:\reviewed\snapshot.json --purpose research_training
```

[Snapshot ingestion](sources/ingestion.py) checks approval **before reading the supplied file**. It does not download anything. PlantVillage/PlantDoc adapters preserve image metadata, rights, provenance and stable leaf/original-image groups; they detect within-snapshot group leakage. No image bytes are loaded, and cross-release split isolation still requires downstream group enforcement.

`adapter-schema` exports the exact machine-readable envelope, columns, nested types, accepted units and limitations for each implemented source. Every snapshot is one UTF-8 JSON envelope containing `schema_version`, `source_id`, `dataset_id`, pinned `source_version`, reviewed `source_url`, timezone-qualified `retrieved_at`, `citations`, `license`, `data_origin`, `format`, and `records`. Unknown fields, ambiguous units and invented missing values are rejected. License metadata includes name, URL, attribution and upstream-rights notes. These declarations are not independent rights verification. Outputs retain original bytes and immutable checksummed lineage; they are not added to the synthetic training/retrieval release automatically.

**YELLOW:** parser mechanics pass on invented test fixtures, but actual ingestion remains blocked (all ten sources disabled; all approvals pending). No real data, training or GPU was used. CPU/peak RAM were not measured. Next: source-specific dataset/version/rights review; never approve a catalog as a blanket dataset.

## Phase 11: SoilGrids / ISRIC snapshots

Use `inspect-adapter`, `adapter-schema`, and `ingest-snapshot` with `--source soilgrids` or `--source isric`. [SoilGrids](sources/soilgrids/__init__.py) and [ISRIC](sources/isric/__init__.py) normalize explicitly described point/profile layers, CRS, depth, units, uncertainty and missingness. Source-specific conversions are listed in the exported schema; unsupported units fail rather than being guessed. Native rasters, WCS downloads and live APIs are not implemented.

**YELLOW:** fixture parser tests pass; actual acquisition/ingestion is blocked by dataset/version and intended-use review. Real soil quality is UNVERIFIED. No external data, model fitting or GPU use; CPU/peak RAM were not measured. Next: approve a specific pinned dataset and provide a reviewed snapshot.

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
