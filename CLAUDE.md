# Aerial Imagery Object Detection — MVP

Freelance engagement. Detect and outline road features in overhead imagery.
This file is the working context. Read `docs/DECISIONS.md` for the reasoning
behind anything here that looks arbitrary — most of it is not.

## Scope

**Four classes.** Road, divider, sidewalk, vehicles.

Fire hydrants and people were in the original brief and have been **deferred**.
At satellite GSD they are physically undetectable: a hydrant is ~0.3 m and
occupies ~1 pixel at 30 cm; a person is ~0.5 m across when viewed from
overhead, so ~2 pixels. No model recovers this. They become viable at 7.5 cm
or better. Do not silently reintroduce them.

"Divider" merges three physically different things at the client's request:
raised concrete median, painted centre line, and guardrail. Expect this class
to score worst. That is a known consequence of the merge, not a bug.

## Two-head architecture

| Head | Classes | Task | Output |
|---|---|---|---|
| Segmentation | road, divider, sidewalk | semantic (`-sem`) | polygon outlines |
| Detection | vehicles | object detection | axis-aligned boxes |

Surface classes are continuous regions with no natural instance boundary —
a bounding box around "road" spans the frame and carries no information.
The client asked for a *curved road outline*, which is a polygon. Semantic
segmentation is the correct task; YOLO26 exposes it as the `-sem` variant.

Vehicles stay as detection. Instance identity matters there and boxes are
cheaper than instance masks.

## Imagery

- **MVP**: drone imagery, 2–5 cm GSD, features clearly visible. Sourced from
  public datasets on Kaggle and Roboflow Universe.
- **Production**: satellite, ~30 cm GSD, delivered as GeoTIFF.

**These do not transfer.** A ~10x resolution gap is a domain shift, not a
scale-up. Production will need a fresh dataset and a fresh training run.
MVP annotations have close to zero reuse value. Say so when it comes up.

## Dataset

~500–750 images, annotated for **both** tasks (one shared corpus, two
annotation passes — not 500–750 per model).

Splits: train / val / benchmark. The benchmark split is **immutable** and
pinned by file hash in `configs/splits/benchmark.txt`. Automated retraining
on a growing dataset silently leaks new images into the test set otherwise.
`harness/gate.py` asserts no overlap at run start; do not disable that check.

Benchmark must sample across source datasets, not concentrate in whichever
one was easiest to label.

### Provenance is mandatory

Every source dataset needs a row in `docs/provenance.csv`: name, URL,
licence, image count, commercial-use permitted, attribution required, GSD.

Kaggle and Roboflow "open source" does not mean commercially usable.
CC BY-NC and CC BY-SA both block a commercial deliverable, and unlicensed
is very common on Roboflow Universe. xView is CC BY-NC-SA and DOTA is
academic-use-only — **neither can be used here** despite appearing in every
aerial-detection reading list. Trained weights inherit the most restrictive
licence in the mix, and you cannot untrain a subset.

## Licensing — unresolved

Ultralytics is dual-licensed AGPL-3.0 / Enterprise. Their licence page states
that fine-tuned models are covered and that AGPL compliance means releasing
the entire derivative work including model weights. Roboflow states plainly
that custom-trained versions are still AGPL-3.0.

**The weights we deliver are themselves encumbered.** The question bites at
delivery, not at production launch. Ultralytics' own public statements are
inconsistent on this — a maintainer has said AGPL permits freelance client
work, while the licence page says otherwise.

Status: awaiting written client decision. Do not assume it is settled.

Permissive alternatives if the answer is no: segmentation-models-pytorch
(MIT) for the surface classes, RT-DETR or RF-DETR (Apache 2.0) for vehicles.
TorchGeo (MIT) wraps SMP and handles GeoTIFF natively — that is the
production path. Check *weights* licences separately from framework
licences; SegFormer code is Apache 2.0 but NVIDIA's weights are
research-only.

## Hardware

**TPUs do not work.** Ultralytics has never supported XLA; the feature
request was closed as not planned. Supported: CUDA, CPU, MPS, Intel XPU,
Ascend NPU. Edge TPU export is quantised TFLite for inference on Coral
devices — unrelated to training.

Free Colab has no API and cannot be triggered from CI. Colab Enterprise
(`gcloud colab executions create`) can, but is billed GCP.

## Conventions

- **Indian/British English** in all prose and documentation: licence (noun),
  optimise, labelled, catalogue.
- **Per-class metrics, always.** Aggregate mAP hides everything that matters
  here. Road dominates pixel count; divider is a rounding error. A gate on
  aggregate metrics will pass models that do not work.
- Config-driven runs. No hardcoded hyperparameters in scripts. Every run is
  reproducible from a committed YAML plus a dataset version.
- Training is a parameterised script, never a notebook.

## Deliverables

Script only — no API, no UI. Client confirmed this for the MVP.

1. Trained weights (both heads)
2. CLI inference script: image in, overlay + JSON out
3. Annotated dataset and annotation guideline
4. Evaluation report with per-class metrics and failure analysis
5. Technical note covering limitations

JSON carries ordered polygon vertices in image pixel coordinates so the
consumer can render directly. Render the overlay **from the polygons**, not
from the raw mask — otherwise the picture and the JSON can disagree.

## Acceptance

The client declined to set an accuracy threshold, calling it too subjective
at this stage. Completion is therefore defined by deliverable, not by score:
the MVP is complete on delivery of the trained models plus a per-class
evaluation report on the held-out benchmark set.

MVP accuracy is not a forecast of production accuracy. The gap is a function
of data volume that has not been funded.

## Not in scope

Depth, elevation, potholes, object orientation, video, frame tracking,
geo-referencing of detections, counting analytics, hosted API, web UI.
All explicitly excluded by client answers. Do not build them speculatively.

## Open items

- Client confirmation of the four-class list (email sent)
- Licence decision
- Whether client can supply imagery to reduce sourcing effort
