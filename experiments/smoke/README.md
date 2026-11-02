# Smoke-run evidence (2026-11-02)

These are complete execution checks, not a benchmark study or a claim that the method outperforms alternatives. No GPU rental was used. CPU PyTorch 2.6.0, seed 42. All hyperparameters, input archive hashes, validation metrics, test metrics and training history are recorded in the JSON files. Local absolute paths identify the inputs actually used; replace them with your own directories when reproducing.

- `sd_crypto_pretrained.json`: full fine tuning of the original local Kronos-mini backbone + continuous-feature adapter + point head, 2 epochs, batch size 16. Existing SD-FMM crypto research split (202/67/68 sequences). No tokenizer is applied to these processed single-channel sequences. The originally publisher-named test arrays are reused under this research partition; this is not an independent official test or zero-shot result.
- `ulb_scratch.json`: tiny randomly initialized Kronos architecture + adapter + point head, 1 epoch, batch size 32. Chronological ULB split, 199364/42719/42724 transactions, 384/56/52 positives. This is a scratch execution check, not a pretrained ULB result. One global stream, not individual customer histories.

Both runs select epoch by validation AP and threshold by validation F1, then evaluate the test set once. Neither run is selected as a preferred method based on its test score. Their numbers are not comparable across datasets or initialization strategies. Full training/validation/test prediction CSVs and checkpoints remain local and are excluded from Git.

A separate synthetic OHLCVA plumbing check loaded both real upstream Kronos-mini and Kronos-Tokenizer-2k weights, trained a sequence head, saved the full checkpoint and reconstructed it offline. Labels there were artificial, so no scientific performance claim is attached to that check.
