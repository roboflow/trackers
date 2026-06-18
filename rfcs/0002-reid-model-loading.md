# RFC 0002: Architecture-agnostic `ReIDModel` loading & preprocessing

- **Status:** Draft (implementation spec)
- **Authors:** Roboflow trackers team
- **Created:** 2026-06-17
- **Depends on:** RFC 0001 (Re-ID feature)
- **Target package version:** `trackers` (current `2.4.0`)
- **License context:** Core package is Apache-2.0; pretrained weights must be license-compatible.

---

## 0. TL;DR

`ReIDModel` is defined by three **independent, swappable axes** — *architecture*,
*weights*, and *preprocessing*. The user selects a model with a single
parameter to `ReIDModel.from_pretrained(...)`, which may be a self-describing
Hugging Face repo, a local path, or a small curated alias. Changing the
backbone, the weights, or the preprocessing is always a parameter change, never
a new class. Preprocessing is **explicit and inspectable** (never hidden inside
the model). Training output round-trips through `save_pretrained` /
`from_pretrained` via a small `reid_config.json`, so a user-trained model
reloads with no manual architecture hint.

**Scaling is via self-describing HF repos, not a registry.** Any model published
with a `reid_config.json` (exactly what `save_pretrained` writes) loads with
`from_pretrained("hf://org/repo")` and **zero registration**. There are two
distinct, deliberately separate extension points:

- The **architecture registry** (`architectures.py`) is a *code-level* extension
  point for backbones we implement (`osnet_*`, `timm:*`, later FastReID). It
  grows only when we add a genuinely new architecture — a single registered
  builder, not a new framework.
- The **model-alias map** (`registry.py`) is a *deliberately tiny* curated layer:
  the shipped default(s) plus adapters for external checkpoints that lack a
  `reid_config.json`. It is a convenience, **never a gate**, and community/user
  models do not go in it.

This spec supersedes the draft `ReIDModel` constructors (`from_pretrained(repo_id,
filename)`, `from_timm`, `from_checkpoint`, `BackboneSpec`) currently on the
`feat/reid-phase1` branch.

---

## 1. Goals / Non-goals

**Goals**

1. One obvious entry point: `ReIDModel.from_pretrained(source, *, architecture=, preprocessing=, device=)`.
2. Architecture-agnostic: OSNet (clean-room) and any `timm` model today; new
   architectures added by registering a builder.
3. Weights from anywhere: curated alias, `hf://repo[/file]`, or local path.
4. Preprocessing is explicit, documented, logged, and serialized.
5. Symmetric `save_pretrained` / `from_pretrained` so user-trained models
   reload self-describingly (the training-fit requirement).
6. Loud failure on architecture/weights mismatch (key-match report).
7. `torch` / `timm` stay optional: importing `trackers.core.reid` must not
   import them; all heavy imports are lazy.

**Non-goals (this RFC)**

- Implementing FastReID SBS (deferred; the registry must make it a drop-in later).
- Implementing the training loop itself (only the save/load contract it targets).
- Changing the evaluator / metrics / dataset loaders (unchanged).

---

## 2. Public API

```python
@classmethod
def from_pretrained(
    cls,
    source: str | None = None,                       # alias | "hf://repo[/file]" | local path/dir
    *,
    architecture: str | "nn.Module" | None = None,   # override; required for a bare weights file
    preprocessing: ReIDPreprocessing | None = None,  # override; else from the resolved model card
    device: str = "auto",
) -> ReIDModel: ...

def save_pretrained(self, directory: str) -> None:   # writes weights.safetensors + reid_config.json
    ...

@property
def preprocessing(self) -> ReIDPreprocessing: ...     # the active, explicit pipeline
```

`architecture` selectors:

- `"osnet_x1_0"`, `"osnet_x0_75"`, `"osnet_x0_5"`, `"osnet_x0_25"` — clean-room OSNet.
- `"timm:<name>"` — any timm model (e.g. `"timm:resnet50"`), built with `num_classes=0`.
- a pre-built `nn.Module` — used as-is.

### 2.1 Canonical call sites

```python
ReIDModel.from_pretrained()                                   # default curated model
ReIDModel.from_pretrained("osnet_x1_0_msmt17")                # curated alias (a pretrained identity)
ReIDModel.from_pretrained("osnet_x1_0_market1501")
ReIDModel.from_pretrained("hf://org/repo")                    # repo carries reid_config.json -> arch known
ReIDModel.from_pretrained("hf://org/repo/osnet.safetensors", architecture="osnet_x1_0")
ReIDModel.from_pretrained("/runs/osnet.pth", architecture="osnet_x1_0")  # bare local checkpoint
ReIDModel.from_pretrained("/runs/my_model_dir")               # save_pretrained output -> self-describing
ReIDModel.from_pretrained(architecture="timm:resnet50")       # ImageNet features, no re-ID weights
```

### 2.2 Resolution algorithm

`from_pretrained` resolves `(architecture, weights, preprocessing)` from `source`
in this order, then applies explicit `architecture`/`preprocessing` overrides:

1. `source is None and architecture is None` → use `DEFAULT_MODEL` alias.
2. `source in ALIASES` → the alias's `ModelCard`.
3. `source` is a directory **or** an `hf://` repo containing `reid_config.json`
   → build a `ModelCard` from that config (architecture self-described; no
   override required).
4. `source` is a bare weights file (`.pth` / `.safetensors`, local or
   `hf://repo/file`) → `architecture` is **required**; raise a clear `ValueError`
   if missing. `preprocessing` defaults to `ReIDPreprocessing()` and is logged.
5. `source is None and architecture is not None` → build the architecture with
   its own pretrained weights (timm ImageNet) or random (OSNet); `weights=None`.

Explicit `architecture=` / `preprocessing=` always override the resolved card.
When the default alias is used and its card has a `domain_warning`, emit it as a
`UserWarning`.

---

## 3. Module layout & responsibilities

```
src/trackers/core/reid/
  preprocessing.py    # ReIDPreprocessing (explicit) + transform builder            [exists]
  weights.py          # resolve_weights() + load_state_dict_into() + KeyReport       [exists]
  architectures.py    # build_architecture(), list_architectures()  (rename of backbones.py)
  registry.py         # ModelCard, ALIASES, DEFAULT_MODEL, resolve_model_card(), config (de)serialize
  osnet.py            # clean-room OSNet (unchanged)
  model.py            # ReIDModel: orchestration only (from_pretrained / save_pretrained / extract_*)
  eval/ …             # unchanged
```

`model.py` orchestrates; it must contain **no** architecture-specific branching.
All torch/timm imports remain lazy (inside functions/builders).

### 3.1 `architectures.py`

```python
def build_architecture(
    architecture: str | "nn.Module",
    *,
    num_classes: int = 0,        # >0 adds a classification head (training); 0 = feature extractor
    pretrained: bool = False,    # load the architecture's OWN pretrained weights (timm ImageNet)
) -> "nn.Module": ...

def list_architectures() -> list[str]: ...   # ["osnet_x0_25", ..., "osnet_x1_0"] (+ "timm:<name>" note)
```

- OSNet: delegates to `build_osnet(variant, num_classes=...)`; `pretrained` is
  ignored (OSNet weights always arrive via the weights axis).
- timm: `timm.create_model(name, pretrained=pretrained, num_classes=num_classes)`.
- `nn.Module`: returned as-is.
- Unknown string → `ValueError` listing valid options.

### 3.2 `registry.py`

```python
@dataclass
class ModelCard:
    architecture: str                 # "osnet_x1_0" | "timm:resnet50"
    weights: str | None               # "hf://…" | path | None
    preprocessing: ReIDPreprocessing
    domain_warning: str | None = None

DEFAULT_MODEL = "osnet_x1_0_msmt17_combineall"

ALIASES: dict[str, ModelCard] = { ... }   # curated catalog (see §6)

def resolve_model_card(source: str) -> ModelCard | None:
    """Return a ModelCard for an alias or a config-bearing dir/repo, else None."""

def load_model_config(directory_or_repo: str) -> ModelCard: ...   # reads reid_config.json
def save_model_config(card: ModelCard, directory: str) -> None: ...# writes reid_config.json
```

`reid_config.json` schema (stable, minimal):

```json
{
  "architecture": "osnet_x1_0",
  "preprocessing": {
    "input_size": [256, 128],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "interpolation": "bilinear",
    "to_rgb": true,
    "normalize_embeddings": true
  }
}
```

`ReIDPreprocessing` gains `to_dict()` / `from_dict()` for this round-trip.

### 3.3 `model.py`

`__init__(self, backbone, device, preprocessing)` (unchanged). `from_pretrained`
implements §2.2 using `resolve_model_card`, `build_architecture`,
`resolve_weights`, `load_state_dict_into`. `save_pretrained(dir)` writes
`weights.safetensors` (via `safetensors.torch.save_model`/`save_file`) and
`reid_config.json` (via `save_model_config`). The inference methods
(`extract_features`, `extract_features_from_paths`) are unchanged from the
current draft (they already honor `preprocessing.to_rgb` /
`normalize_embeddings`).

---

## 4. Preprocessing (explicit) — already implemented, keep as-is

`ReIDPreprocessing` (`preprocessing.py`) is a frozen dataclass with
`input_size`, `mean`, `std`, `interpolation`, `to_rgb`, `normalize_embeddings`,
plus `describe()` and `build_transform()`. It is logged on model construction
and documented field-by-field. **Add** `to_dict()` / `from_dict()` for config
serialization (§3.2). No behavioural change.

---

## 5. Weights loading — already implemented, keep as-is

`weights.py` provides `resolve_weights(source)` (local path / `hf://repo[/file]`
with optional `@revision`) and `load_state_dict_into(module, path, device, …)`
returning a `KeyReport`. Loading matches by name + shape, strips `module.`
prefixes, drops classifier heads, and **warns loudly** when the matched fraction
is below threshold (mismatched architecture/weights). `.safetensors` and `.pth`
are both supported. No change beyond what exists.

---

## 6. Model-alias map (tiny curated layer — NOT the scaling path)

The way to scale to arbitrary models is a **self-describing HF repo** (§2.2
step 3): publish `weights.safetensors` + `reid_config.json` and load with
`from_pretrained("hf://org/repo")` — **no registration, no code change**. This is
exactly what `save_pretrained` produces. The alias map is **not** a catalog you
add every model to; it is a deliberately small convenience.

`ALIASES` maps a curated **pretrained identity** to a `ModelCard`. Phase 1 ships
exactly one entry:

| alias | architecture | weights | preprocessing | domain_warning |
|---|---|---|---|---|
| `osnet_x1_0_msmt17_combineall` (**default**) | `osnet_x1_0` | `hf://kaiyangzhou/osnet/…combineall…256x128….pth` | standard 256×128 | yes (pedestrian) |

The alias map exists for only two narrow jobs:

1. **A no-arg default** — `from_pretrained()` needs one `ModelCard` describing the
   default model. Keeping it as a named alias also lets the explicit, reproducible
   form `from_pretrained("osnet_x1_0_msmt17_combineall")` resolve through the same
   path (encourage explicit strings in experiment configs over no-arg magic).
2. **Adapting config-less external checkpoints** — the default weights live in a
   third-party repo with **no** `reid_config.json`; a bare `.pth` cannot state its
   architecture or preprocessing. The card records that `(architecture, weights,
   preprocessing)` triple once so users (and we) don't re-guess preprocessing,
   which silently wrecks accuracy if wrong.

Rules / non-goals:

- Community and user-trained models are **self-describing HF repos**, not aliases.
- The default stays **combineall** (best general-purpose pedestrian features) per
  the maintainer decision. It trains on MSMT17 test identities, so it must **not**
  be used to benchmark MSMT17 — documented on the alias and in the eval notebook.
- Same-domain eval checkpoints (`osnet_x1_0_market1501`, standard-split
  `osnet_x1_0_msmt17`) stay **explicit** in the eval notebook via
  `from_pretrained(path, architecture="osnet_x1_0")` — they are **not** aliases.
- Adding FastReID later = one new architecture in `architectures.py` (the code
  extension point); a published FastReID model is then just an `hf://` repo, with
  an alias added only if we ship/validate a specific config-less checkpoint.

---

## 7. Training fit (forward-compatible)

The future training module (RFC 0001 "later releases") builds the architecture
**with** a classifier head and saves through this contract:

```python
backbone = build_architecture("osnet_x1_0", num_classes=num_train_ids)   # head for training
# … train …
model = ReIDModel(backbone, device, ReIDPreprocessing())
model.save_pretrained("runs/my_reid")        # weights.safetensors + reid_config.json
# later, anywhere:
model = ReIDModel.from_pretrained("runs/my_reid")   # arch + preprocessing from config; head dropped
```

The classifier head is present only during training; `load_state_dict_into`
drops `classifier.*` on load, so the embedding weights round-trip exactly and
inference never depends on `num_classes`. This is why the structure fits training
without special cases.

---

## 8. Back-compat & migration

This is an unreleased feature branch, so we change the API freely:

- **Remove** public `from_timm` and `from_checkpoint` classmethods (replaced by
  `from_pretrained(architecture="timm:…")` and `from_pretrained(path,
  architecture=…)`). Keep their logic as **internal** helpers where it aids clarity.
- **Rename** `backbones.py` → `architectures.py`; `BackboneSpec` → `ModelCard`
  (moved to `registry.py`).
- Update `src/trackers/core/reid/__init__.py` exports accordingly
  (`ReIDPreprocessing`, `ModelCard`, `list_architectures`, `resolve_weights`,
  `KeyReport`).
- Update both notebooks (`notebooks/eval_reid.ipynb`,
  `notebooks/eval_trackers_reid.ipynb`) to the new API:
  - `ReIDModel.from_pretrained()` (default) stays valid.
  - `ReIDModel.from_checkpoint(p)` → `ReIDModel.from_pretrained(p, architecture="osnet_x1_0")`.
- `extract_features(detections, frame)` and
  `extract_features_from_paths(paths, batch_size, normalize)` signatures are
  **unchanged** (tracking path + evaluator depend on them).

---

## 9. Minimal test plan

Keep tests **minimal and legible** (mirror existing `tests/core/` style). Guard
torch/timm-dependent tests with `pytest.importorskip`. Target ~5–6 tests total in
`tests/core/reid/`:

1. **preprocessing** — `describe()` is non-empty; `build_transform()` returns a
   callable; unknown `interpolation` raises `ValueError`; `to_dict()`/`from_dict()`
   round-trip.
2. **resolution** — `resolve_model_card("osnet_x1_0_msmt17_combineall")` returns a
   card; unknown alias → `None`; bare-path source without `architecture` raises a
   clear `ValueError`.
3. **weights** — `resolve_weights` raises `FileNotFoundError` (missing path) and
   `ValueError` (malformed `hf://`); `load_state_dict_into` reports 100% on a
   matching module and **warns** on a mismatched one.
4. **model smoke** (torch) — `from_pretrained(architecture="timm:resnet18",
   ...)` (or a tiny `nn.Module`) → `extract_features` returns `(N, D)`,
   L2-normalised; empty detections → `(0, 0)`.
5. **round-trip** (torch) — `save_pretrained(tmp)` then
   `from_pretrained(tmp)` rebuilds the same architecture + preprocessing and
   produces identical embeddings.

No network in tests (no real checkpoint downloads).

---

## 10. Implementation task breakdown (for delegated agents)

Ordered; later tasks depend on earlier ones.

- **T1 — architectures.py**: rename/rework `backbones.py` into `architectures.py`
  with `build_architecture` (+`num_classes`,`pretrained`) and `list_architectures`.
- **T2 — registry.py**: `ModelCard`, `ALIASES`, `DEFAULT_MODEL`,
  `resolve_model_card`, `load_model_config`/`save_model_config`; add
  `ReIDPreprocessing.to_dict/from_dict`.
- **T3 — model.py**: implement §2 (`from_pretrained` resolution + overrides +
  domain warning + key-report logging, `save_pretrained`); remove `from_timm` /
  `from_checkpoint`; keep inference methods.
- **T4 — exports**: update `reid/__init__.py`.
- **T5 — notebooks**: migrate both notebooks to the new API.
- **T6 — tests**: the minimal suite in §9.
- **T7 — docs**: reconcile RFC 0001 OSNet target numbers to the validated
  model-zoo figures (MSMT17 standard split ≈ 74.9 R1 / 43.8 mAP) and note the
  combineall-default caveat.

**Acceptance:** `uv run pytest tests/core/reid -q` passes; `python -c "import
trackers.core.reid"` works **without** torch/timm installed; ruff/lint clean.
