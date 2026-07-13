# Contrastive Steering v2

This folder contains five standalone steering experiments for the multiplication task (`y = a*b + c`).

Each script:

- fits `TabPFNRegressor` on the multiplication dataset,
- constructs matched multiplicative/additive eval batches (`N_EVAL=32`),
- computes a contrastive direction `delta = mean(mult) - mean(add)`,
- sweeps `alpha` from `0.0` to `10.0` (step `0.5`),
- saves one plot under `docs/attention_head_patching/`,
- saves one JSON file under `results_contrastive_steering_v2/`.

## Scripts

- `steering_full_mha_output.py` (A): full 192-d `self_attn_between_features` output at Layer 0
- `steering_residual_layer0.py` (B): residual stream output at Layer 0 (`PerFeatureEncoderLayer.forward`)
- `steering_full_mha_layer6.py` (C): full 192-d `self_attn_between_features` output at Layer 6
- `steering_target_token.py` (D): target token residual slice only (feature block index 3) at Layer 0
- `steering_items_attention.py` (E): full 192-d `self_attn_between_items` output at Layer 0

## Run order

1. `python src/attention/steering/steering_full_mha_output.py`
2. `python src/attention/steering/steering_residual_layer0.py`
3. `python src/attention/steering/steering_full_mha_layer6.py`
4. `python src/attention/steering/steering_target_token.py`
5. `python src/attention/steering/steering_items_attention.py`
