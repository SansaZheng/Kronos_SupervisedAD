# Supervised implementation validation — 2026-11-02

This is an implementation and execution record, not a completed effectiveness study.

## Source and tests

- Vendored `model/` exactly matches upstream Kronos commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a`; see UPSTREAM.md for all three Git blob hashes.
- CPU environment: Python 3.12, PyTorch 2.6.0; test runtime dependencies include numpy, pandas, scikit-learn 1.7.2, pytest, einops, huggingface_hub and safetensors. JSON reports record the actual Torch version and all run arguments.
- `python -m pytest tests -q`: **33 tests** covering original data integrity and new model/training invariants.
- Model tests include causal future perturbations, padding, gradient propagation through a frozen backbone, ignored unknown labels, train-only statistics, input schema/target-column checks, split overlap, preservation of missing-time gaps and timestamp ties, token-path reconstruction, point/sequence training and offline prediction.
- In the end-to-end leakage tests, replacing every test label leaves the selected epoch, validation threshold, validation metrics and every checkpoint model tensor unchanged. Setting prediction labels to unknown also leaves prediction scores unchanged.
- Original data validators verify 20 source files, 5 prepared artifacts, 9 transaction assets / 25 parts and original decompressed source hashes. No raw labels or source arrays were changed by this implementation.

Run the complete checks:

```bash
python -m pip install -r requirements-supervised.txt
python -m pytest tests -q
python scripts/data/validate_supervised_data.py
python -m scripts.data.transaction_data verify --source-bytes
```

## Actual executions

Full machine-readable records: [SD-FMM crypto, pretrained](../experiments/smoke/sd_crypto_pretrained.json) and [ULB, scratch](../experiments/smoke/ulb_scratch.json).

| Execution | Initialization and adaptation | Training | Test AP | Test positive F1 | Scope |
| --- | --- | --- | ---: | ---: | --- |
| SD-FMM crypto | Original Kronos-mini pretrained backbone, continuous adapter, point head; full backbone fine tuning | 2 epochs, seed 42, batch 16 | 0.185374 | 0.264 | 68 sequences, 10,880 points, 68 positive points |
| ULB | Tiny random Kronos architecture, continuous adapter, point head | 1 epoch, seed 42, batch 32 | 0.184310 | 0.250 | 42,724 transactions, 52 positives; chronological holdout |

Each run chooses the epoch by validation AP and the threshold by validation F1 before scoring test data. These two runs are on different datasets with different initialization; they are **not a comparison between pretraining and scratch**. Neither is a systematic hyperparameter search or a comparison against strong baselines. Low epoch counts are deliberate execution checks. SD-FMM's repository research partition reuses publisher-named test arrays and must not be reported as the original paper's independent or zero-shot test.

The pretrained run also reloads its checkpoint without downloading model weights, and reproduces all 10,880 test probabilities. A separate OHLCVA plumbing run loaded both original local Kronos-mini and Kronos-Tokenizer-2k, trained a sequence head for one epoch, and reconstructed its complete checkpoint offline; that input used synthetic prices and artificial labels, so its classification metrics are not evidence of detection quality.

Successful data preparation also checked full Sapienza 15S (327103 / 111451 / 145550 labelled observations; 190 / 50 / 77 positive starts), TSTR (577 / 193 / 194 sequences), SD-FMM stock (16 / 5 / 6 sequences), and SD-FMM crypto (202 / 67 / 68 sequences). Sapienza's observed-point windows preserve real gaps via a time-gap feature and do not interpolate missing blocks.

## Remaining research work

Train and compare against Random Forest / boosting and ordinary causal sequence classifiers using the same splits; repeat seeds and report uncertainty. Compare pretraining, freezing and scratch on each same dataset. Evaluate tokenizer preprocessing and continuous-adapter distribution shifts. Establish task-specific causal preprocessing for categorical transaction datasets and graphs before including them in experiments. No claim of state-of-the-art performance, cross-market generalization, profitability, or deployment readiness is made.
