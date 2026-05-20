# Gemma3 transcoder ids (resolved during Task A2)

- Gemma3-1B-it: `mwhanna/gemma-scope-2-1b-it/transcoder_all/width_16k_l0_small_affine`
- Gemma3-4B-it: `mwhanna/gemma-scope-2-4b-pt/transcoder_all/width_16k_l0_small_affine` (only `-pt` found in tests; README confirms IT variants exist in the mwhanna collection — check https://huggingface.co/collections/mwhanna/gemma-scope-2-transcoders-circuit-tracer for `mwhanna/gemma-scope-2-4b-it/...`)
- Gemma3-4B-pt (confirmed): `mwhanna/gemma-scope-2-4b-pt/transcoder_all/width_16k_l0_small_affine`

Source: `/home/hyunjin/circuit-tracer/tests/test_attributions_gemma3_nnsight.py:533-534` (1B-it confirmed);
        `/home/hyunjin/circuit-tracer/tests/test_offload.py:83` and `test_freeze_points_hessian.py:39` (4B-pt confirmed)
        README.md line 50 states "PT and IT" variants available for 270M, 1B, 4B, 12B, 27B

Note: The existing `configs/gemma3_4b_config.json` uses `google/gemma-scope-2-4b-pt` as a placeholder;
the actual working HF path format is `mwhanna/gemma-scope-2-<size>-<variant>/transcoder_all/<width>`.

Smoke load: PASS (9.7s, no OOM, model.backend=='nnsight', cfg.n_layers>0, tokenizer loaded)
