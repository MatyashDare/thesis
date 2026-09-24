# Context-Aware NLU for the MBD Dataset

Joint **intent detection** and **slot filling** over multi-turn in-car dialogues,
built on a context-aware CHAN encoder over a multilingual transformer, with a CRF
slot decoder and focal loss.

Best model: **86.8 % intent top-1** (190 classes) and **81.1 % slot span F1** on the
held-out MBD_augment test split.

---

## Table of contents

- [Results](#results)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Inference demo](#inference-demo)
- [Training](#training)
- [Evaluation](#evaluation)
- [Evaluation protocol](#evaluation-protocol)
- [Datasets](#datasets)
- [Model architecture](#model-architecture)
- [Experiment artifacts](#experiment-artifacts)

---

## Results

### MBD_augment (test split, corrected protocol)

| Experiment | Encoder | Optimiser | Intent top-1 | Intent macro-F1 | Top-3 | Slot span F1 | Latency |
|---|---|---|---|---|---|---|---|
| **`mbd_aug_basebert_adamw_cosine_lr3e5_e20`** | mBERT | AdamW + cosine | **0.8678** | **0.8624** | **0.9566** | 0.8107 | 13.4 ms |
| `mbd_aug_mdeberta_lion_cosine_lr2e5_e20` | mDeBERTa-v3 | Lion + cosine | 0.8485 | 0.8420 | 0.9269 | **0.8261** | 25.2 ms |
| `mbd_aug_basebert_adamw_cosine_lr2e5_e20` | mBERT | AdamW + cosine | 0.3220 † | – | – | 0.7738 † | 13.5 ms |

† Validation-only; this run under-converged on intents and is retained solely as a
learning-rate ablation (2e-5 vs 3e-5 on an otherwise identical configuration).

**mBERT wins on intents, mDeBERTa wins on slots.** mDeBERTa is also ~1.9x slower per
utterance, so mBERT is the better default and is what the demo loads.

### Other datasets

| Dataset | Experiment | Intent ‡ | Slot F1 ‡ |
|---|---|---|---|
| SGD | `cw5_both` | 0.9908 | 0.9575 |
| SGD | `cw3_both` | 0.9906 | 0.9543 |
| MBD (un-augmented) | `mbd_aug_fullctx_focal_crf_e20` | 0.7235 | 0.7988 |

‡ Legacy per-batch-averaged metrics, kept as historical curves. They are *not*
directly comparable to the MBD_augment figures above, which use the corrected
protocol. See [Evaluation protocol](#evaluation-protocol).

The un-augmented MBD run scores 0.7235 against MBD_augment's 0.8678 on the same
architecture, which is the clearest single measure of what augmentation bought.

---

## Repository structure

```
thesis/
├── demo.py                  # ← interactive inference (start here)
├── train.py                 # training / testing entry point
├── evaluate.py              # corrected re-evaluation of a checkpoint
├── run_experiments.py       # multi-GPU experiment series
├── _bootstrap.py            # puts src/ on sys.path
│
├── data/                    # datasets
│   ├── MBD/                 #   original corpus
│   ├── MBD_augment/         #   augmented corpus (used for all new experiments)
│   └── SGD/                 #   → reference/CaBERT-SLU/data/sgd_dialogue
│
├── experiments/             # best 3 runs per dataset
│   ├── MBD_augment/
│   │   ├── mbd_aug_basebert_adamw_cosine_lr3e5_e20/   ← best overall
│   │   ├── mbd_aug_mdeberta_lion_cosine_lr2e5_e20/
│   │   ├── mbd_aug_basebert_adamw_cosine_lr2e5_e20/
│   │   └── mbd_augment_series/                        ← cross-run summary
│   ├── MBD/
│   └── SGD/
│
├── src/nlu/                 # the library
│   ├── config.py            #   CLI + path resolution
│   ├── paths.py             #   canonical repo locations
│   ├── utils.py             #   IntentEvaluator / SlotEvaluator
│   ├── data/                #   corpus loading, tokenisation, context batching
│   ├── models/              #   CHAN encoder, CRF, transformer blocks
│   ├── training/            #   train/test loops, experiment series
│   ├── evaluation/          #   standalone evaluation + metric migration
│   ├── visualization/       #   curves, confusion plots
│   └── inference/           #   predictor, ICE markup, interactive session
│
├── scripts/                 # thin shell wrappers
├── tools/                   # dataset augmentation
└── reference/CaBERT-SLU/    # upstream baseline, kept for provenance
```

---

## Installation

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

Everything below assumes the interpreter at `./venv/bin/python`.

---

## Inference demo

The demo starts a dialogue session in your terminal. Type a request, and the model
answers with the predicted intent (plus a diagnostic top-3) and the predicted slots
rendered as **ICE markup**. Keep typing to continue the dialogue — earlier turns are
used as context — and type `exit` to end it.

```bash
./venv/bin/python demo.py
# or
./scripts/demo.sh
```

```text
==========================================================================
               Context-aware NLU - interactive session (MBD)
==========================================================================
 experiment : mbd_aug_basebert_adamw_cosine_lr3e5_e20
 encoder    : bert-base-multilingual-cased
 intents    : 190     slot labels: 348
 context    : on (both, window=full dialogue)
 ICE values : lexicon (5409 entries)
--------------------------------------------------------------------------
 Type a request. 'help' for commands, 'exit' to end the session.
==========================================================================

you> set the fan to maximum on the passenger side
  Intent      : set.fan  (96.3%)
  Top-3       : 1. set.fan (96.3%)  2. set.seatVentilation (2.3%)  3. increase.fan (0.6%)
  ICE markup  : set the fan to <maximum>[setting=maximum] on the <passenger>[area=passenger] side
  Slots       : setting=maximum, area=passenger

you> now make it quieter
  Intent      : decrease.volume  (93.7%)
  Top-3       : 1. decrease.volume (93.7%)  2. increase.volume (1.7%)  3. skip.sequenceRewind (0.9%)
  ICE markup  : now make it quieter
  Slots       : (none)

you> navigate to Sandhurst
  Intent      : navigate.location  (98.5%)
  Top-3       : 1. navigate.location (98.5%)  2. navigate.locationEmbedded (1.2%)  3. navigate.waypoint (0.1%)
  ICE markup  : navigate to <Sandhurst>[city=Sandhurst]
  Slots       : city=Sandhurst

you> exit
Session ended; context discarded.
```

### Session commands

| Command | Effect |
|---|---|
| `exit` / `quit` | End the session. Context is discarded; nothing is persisted. |
| `reset` | Clear the dialogue context but stay in the session. |
| `context` | Report how many turns are currently in context. |
| `/sys <text>` | Add a *system* turn to the context without predicting. |
| `help` | Show the command list. |

### Options

```bash
python demo.py --text "fan passenger side to maximum"   # one-shot, then exit
python demo.py --json                                   # JSON lines output
python demo.py --no-context                             # every turn independent
python demo.py --no-lexicon                             # ICE values = surface text
python demo.py --top_k 5                                # widen the diagnostic list
python demo.py --experiment_dir experiments/MBD_augment/mbd_aug_mdeberta_lion_cosine_lr2e5_e20
```

With no `--experiment_dir`, the demo auto-selects the MBD_augment experiment with the
highest `best_joint_score`.

### ICE markup

Slots are rendered in the corpus's own inline format, `<surface>[type=value]`:

```text
fan <passenger side>[area=passenger] to <maximum>[setting=maximum]
```

The tagger predicts the slot **type** for each token, but the **value** is a corpus
normalisation rather than a function of the surface form — `"right down"` normalises to
`entirely`, `"a bit"` to `small_change`. So a surface→value lexicon (5 409 entries over
170 slot types) is mined from the training split, letting the demo reproduce gold
annotations exactly. Unseen phrases fall back to their surface text. Use
`--no-lexicon` to disable this and see the raw tagger output.

### Programmatic use

```python
from nlu.inference import ContextualNLUPredictor, ValueLexicon

predictor = ContextualNLUPredictor(
    experiment_dir="experiments/MBD_augment/mbd_aug_basebert_adamw_cosine_lr3e5_e20",
    data_dir="data/MBD_augment",
    lexicon=ValueLexicon.from_dataset("data/MBD_augment"),
)

prediction = predictor.predict("navigate to Sandhurst")
print(prediction.intent)     # navigate.location
print(prediction.slots)      # ['city=Sandhurst']
print(prediction.ice)        # navigate to <Sandhurst>[city=Sandhurst]
print(prediction.as_dict())  # JSON-ready

predictor.reset()            # drop dialogue context
```

The first launch derives the label space from the corpus and caches it to
`label_space.json` inside the experiment directory; later launches start in ~14 s.

---

## Training

```bash
./scripts/train_mbd_augment.sh my_run
```

or explicitly:

```bash
./venv/bin/python train.py \
  --mode train \
  --data_dir data/MBD_augment \
  --experiment_name my_run \
  --bert_model_name bert-base-multilingual-cased \
  --epochs 20 --batch_size 32 --maxlen 36 \
  --context_mode both --context_window_size -1 \
  --intent_loss focal --focal_gamma 2.0 \
  --slot_loss_weight 1.5 --use_slot_crf \
  --optimizer adamw --lr_scheduler cosine \
  --lr_bert 3e-5 --lr_head 5e-4 \
  --warmup_ratio 0.1 --grad_clip 1.0 --seed 0
```

Results land in `experiments/<dataset_group>/<experiment_name>/`, where
`dataset_group` defaults to the basename of `--data_dir`.

Supported knobs: encoders (`bert-base-multilingual-cased`, `microsoft/mdeberta-v3-base`),
optimisers (`adamw`, `lion`, `adafactor`, `adam`, `sgd`), schedulers (`cosine`, `linear`,
`plateau`, `step`, `none`), intent losses (`focal`, `ce`, `bce`, weighted variants), and
CRF or softmax slot decoding.

### Experiment series

```bash
./venv/bin/python run_experiments.py                 # train + test the full series
./venv/bin/python run_experiments.py --summary-only  # rebuild summary + conclusion
```

Runs are distributed across GPUs (`MBD_EXPERIMENT_GPU_COUNT`, default 2) and produce a
ranked cross-run summary in `experiments/MBD_augment/mbd_augment_series/`.

---

## Evaluation

Re-score any checkpoint with the corrected protocol:

```bash
./scripts/evaluate.sh experiments/MBD_augment/mbd_aug_basebert_adamw_cosine_lr3e5_e20
# or, for a different encoder:
./venv/bin/python evaluate.py \
  --experiment_dir experiments/MBD_augment/mbd_aug_mdeberta_lion_cosine_lr2e5_e20 \
  --data_dir data/MBD_augment \
  --bert_model_name microsoft/mdeberta-v3-base \
  --splits validation,test
```

This writes per-split metrics, a per-class intent breakdown, a confusion table, a
per-slot-type breakdown, a text report and confusion plots into
`<experiment>/evaluation/`.

---

## Evaluation protocol

Intent scoring is **strictly single-label**: the model emits exactly one intent per turn
via `argmax` over the logits. There is no sigmoid and no multi-label thresholding. The
top-k list shown by the demo is diagnostic only and never changes the prediction.

A consequence worth stating explicitly, because it is a frequent source of confusion:

> For single-label classification, **micro-P = micro-R = micro-F1 = top-1 accuracy**.

So a single number describes intent performance; macro-F1 (unweighted over the 190
classes) and weighted-F1 are reported alongside it to expose rare-class behaviour.

Slots are scored as **BIO spans with strict boundary + type agreement**, computed
**per utterance** so spans cannot bleed across turn boundaries, with `[CLS]`, `[SEP]`
and padding excluded from token accuracy.

All metrics accumulate gold/predicted arrays over the **entire split** and are converted
to scores once, at the end.

### Metrics that were corrected

Earlier runs in this repository used a metrics implementation with four defects. They
are fixed in [`src/nlu/utils.py`](src/nlu/utils.py); legacy CSVs are preserved as
`metrics_legacy_raw.csv` and flagged with a `metric_note` column.

1. **Intent accuracy was scaled by ~157x.** Per-batch accuracy *fractions* were summed
   and then divided by the number of *turns* rather than the number of *batches*, so a
   true 0.86 was reported as 0.0055.
2. **Macro-F1 was averaged per batch.** A 32-sample batch can contain at most 32 of the
   190 classes, so per-batch macro-F1 is not a meaningful quantity, and averaging it
   across batches does not recover the true value.
3. **Slot spans bled across utterances.** The batch was flattened into a single tag
   stream, so an entity ending one utterance merged with one beginning the next.
4. **Padding contaminated token accuracy.** `[PAD]` labels assigned to `[CLS]`/`[SEP]`
   and padding positions were counted as correct predictions.

Both `evaluate.py` and `train.py --mode test` implement the protocol independently and
produce identical numbers, which is used as a cross-check.

### Why intents are not "badly detected"

The dataset has **190 intent classes**. Measuring the label noise directly: only 2.6 %
of utterances are ambiguous (identical text mapped to more than one intent), giving an
**oracle ceiling of 98.66 %** on top-1 accuracy. The ambiguous cases are genuine
context-dependent follow-ups such as *"sort them by date, please"* — precisely what the
context encoder exists to resolve, not annotation errors.

The remaining errors are concentrated in semantically adjacent pairs
(`read.currentlyPlayingInformation` → `read.trackInformation`,
`start.offlineEntertainmentSource` → `select.offlineEntertainmentSource`), which is why
top-3 accuracy jumps to 95.7 %. Per-class and confusion breakdowns are in each
experiment's `evaluation/` folder.

---

## Datasets

| Directory | Turns | Content |
|---|---|---|
| `data/MBD` | — | Original in-car dialogue corpus, `en-GB` + `de-DE` |
| `data/MBD_augment` | 166 458 | Augmented corpus — all current experiments use this |
| `data/SGD` | — | Schema-Guided Dialogue (upstream CaBERT-SLU baseline) |

MBD_augment splits into 151 657 train / 5 190 validation / 9 611 test turns across 190
intents and 348 slot labels, with **no unseen labels** in validation or test.

Rows carry the inline `annotated_utterance` (ICE markup), the plain `utterance`, derived
`bio_slots`, and dialogue/turn identifiers that define the context structure.

Regenerate the augmented corpus with:

```bash
./venv/bin/python tools/create_augmented_dataset.py
```

Augmentation combines LLM-synthesised paraphrases, date/time expansion, cross-dialogue
mixing, and rare-intent oversampling.

---

## Model architecture

```
utterance tokens ──► transformer encoder (mBERT / mDeBERTa-v3)
                          │
                          ├──► CHAN context attention over dialogue history
                          │         │
                          │         ├──► intent classifier ──► focal loss ──► argmax
                          │         │
                          └─────────┴──► slot BiRNN + CRF ──► Viterbi decode
```

- **Context** — `--context_mode both` feeds user *and* system turns;
  `--context_window_size -1` uses the full dialogue.
- **Focal loss** — counteracts the heavy class imbalance across 190 intents.
- **CRF** — enforces valid BIO transitions instead of independent per-token argmax.
- **Intent-conditioned slots** — the slot head receives the intent context vector, so
  the two tasks are genuinely joint rather than merely sharing an encoder.

---

## Experiment artifacts

Each experiment directory contains:

| Path | Contents |
|---|---|
| `experiment_setup.json` | Full configuration, hyper-parameters, and final metrics |
| `metrics.csv` | Per-epoch training curve (corrected schema) |
| `metrics_legacy_raw.csv` | Original pre-correction curve, kept for provenance |
| `train_report.txt`, `experiment_report.txt` | Human-readable summaries |
| `evaluation/` | Corrected per-split metrics, per-class + confusion CSVs, plots |
| `test/` | Test-time outputs including per-utterance latency/memory |
| `training_summary.png`, `*_losses.png` | Loss and metric curves |
| `logs/` | Training and test logs |
| `label_space.json` | Cached intent/slot label space for fast inference startup |

Cross-run artifacts live in `experiments/MBD_augment/mbd_augment_series/`:
`series_summary.csv`, `series_conclusion.txt`, `all_experiment_setups.json`.

---

## Known limitations

- **Slot macro-F1 (0.62) trails span micro-F1 (0.81)** — rare slot types are weak, and
  this is the clearest remaining target for improvement.
- **Validation loss rises after epoch 3** while accuracy keeps climbing: standard
  over-confidence under focal loss. Best-epoch checkpointing handles it, but simply
  training longer will not help.
- The 0.90 intent target is not yet met (0.8678). The oracle ceiling of 98.66 % shows
  the headroom is real; ensembling and rare-class slot work are the natural next steps.
