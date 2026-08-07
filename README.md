# ML_ZK

Research code for studying **machine-learning execution traces**, **activation reconstruction / separation**, and **soundness attacks** in settings relevant to ZK-style verification of neural network (and LLM) inference.

Authored for public documentation under **HSKayman**. Large artifacts (`.pth`, most `.csv`, `.npy`, and most `.md`) are gitignored; this repo focuses on experiment scripts and notebooks.

## What this repo studies

- **Trace separation (TS):** whether two models (or calculated vs hooked activations) produce distinguishable internal traces on shared inputs.
- **Completeness / formula validation:** whether reconstructed layer outputs match the model’s real activations (CNN/ResNet and Llama-2).
- **Soundness attacks:** whether an adversary can force malicious outputs or activations (swap, malicious input, neuron/delta perturbation, reverse-transform, gradient attacks) while looking consistent under naive checks.
- **ZK-LLM probes:** early Llama-2-7B attack and verification notebooks exploring activation/gradient gaps and inverse-transform style attacks.

## Top-level layout

| Directory | Role |
|-----------|------|
| `TraceSeperationCustomCNN/` | Custom CNN train/test/visualize pipeline for activation-hook trace separation. |
| `TraceSeperationResnet/` | ResNet-based trace-separation train/test/visualize. |
| `TraceSeperationResnetv2/` | Refined ResNet trace-separation variant (+ local `data4model_*` trees). |
| `TraceSeperationResnetv3/` | Latest ResNet trace-separation variant (+ local `data4model_*` trees). |
| `TraceSeperationLlama/` | Llama-2 TS3 tracing, weight analysis, and result checking/visualization. |
| `SwapAttackLlama/` | Logit/activation **swap attacks** on Llama with analyzers and visualizers. |
| `MaliciousInputAttackLlama/` | **Malicious-input** attacks on Llama with hook-based activation comparison. |
| `MaliciousInputAttackANN/` | Gradient-based malicious-input attacks on small ANNs (notebooks). |
| `PerturbAttackForNeuron/` | Neuron-level perturbation / epsilon swap attacks, result plots, z-value scraping. |
| `PerturbAttackForDelta/` | Delta/norm-focused perturbation attacks and CSV extraction helpers. |
| `Archive/` | Historical experiments, Finalize completeness/soundness packs, and early ZK-LLM work (see below). |

## Active experiment themes

### Trace separation (vision + LLM)

CNN/ResNet folders typically share a pattern:

- `model_structure.py` — model factory, preprocessing, train/eval helpers  
- `trainer.ipynb` — train models for a dataset split  
- `tester.py` / `tester.ipynb` — register hooks, sample neurons, compare activations across models  
- `visualizer.py` — divergence plots (e.g. JS divergence)

`TraceSeperationLlama/` applies the same idea to Llama-2 (`LLM-TS3.*`, weight analyzers, result checkers).

### Attacks on Llama / ANN

- **SwapAttackLlama** — craft malicious logits / swaps; compare reconstructed vs real layer outputs.  
- **MaliciousInputAttack\*** — search inputs that induce target activation or decision behavior.  
- **PerturbAttackForNeuron / PerturbAttackForDelta** — edit neurons under epsilon/delta budgets to flip or swap predictions; includes ranking, norms, and plotting utilities.

## Archive overview

`Archive/` keeps earlier iterations and paper-oriented Finalize bundles. Source is annotated; bulk datasets under Archive are not the documentation target.

| Area | Contents |
|------|----------|
| `Experiment_1_GrayScaled` … `Experiment_4_RGB_PTR` | Early CNN/ResNet grayscale→RGB pipelines (preprocess, train, activation/adversarial testers). |
| `Experiment_5_OPT_VIC` | Optimizer/victim style ANN experiments. |
| `Experiment_6_BG` / `_v2` | Toy ANN demos: activation attack, input inversion, reverse-transform (ReLU/sigmoid), unlearning. |
| `Experiment_7_VerifyFormula` | Small ANN formula-verification notebook. |
| `Analysis of Softmax` | Llama logit/softmax analysis, swap-attack scripts, TS3 single-neuron testers. |
| `Finalize/Completeness (Formula Validation)` | CNN + Llama-2 completeness/formula checks. |
| `Finalize/Soundness (Attacks)` | ANN + Llama-2 soundness attack notebooks/scripts. |
| `Finalize/SQ1`–`SQ3` | Focused soundness-question attack/visualizer pairs. |
| `Finalize/TS1`–`TS3` | Packaged trace-separation (CNN/ResNet + Llama) Finalize copies. |
| `ZK_LLM_attack` / `ZK_LLM_attack_2` | Early Llama-2-7B ZK-LLM attack and inverse-transform notebooks. |
| `ZK_LLM_verify` / `ZK_LLM_verify_2` | Early Llama-2 ZK verification / probing notebooks. |
| `SwapAttackInjectLlama` | Injected neuron/delta swap attacks + boundary analysis. |
| `Dog Breeds` | Dataset organization helper for breed-structured image folders. |
| `raw_data` | Local raw image classes (cat/dog/squirrel); typically data-heavy / not source docs. |

## Notes

- Prefer the **non-Archive** directories for the current attack and trace-separation workflows; use `Archive/` for provenance and earlier experiment variants.
- Model weights (`.pth`), NumPy dumps (`.npy`), and most CSVs are ignored by `.gitignore` (with a few explicit CSV allowlists). Clone + re-run training/tests as needed to regenerate artifacts.
- Source files (`.py`, `.ipynb`) include a short top-of-file description of their role.
