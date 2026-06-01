# Retail SFT / async-GRPO submission

`submit_sft.py` reproduces the Foundry job that was originally submitted ad-hoc
in chat (`zv-rft-32b-sft-h-mf2y` on H100, `zv-rft-32b-sft-a-3nwl` on A100).
The payload was recovered by GET-ing the running job via the Foundry REST API.

## Usage

```bash
# H100 (testfoundrywcusclustergpu, 4 nodes, ND96r_H100_v5)
python submit_sft.py --cluster h100

# A100 (testfoundrywcusclustera100, 4 nodes, ND96amsr_A100_v4)
python submit_sft.py --cluster a100

# Explicit name
python submit_sft.py --cluster h100 --name zv-rft-32b-sft-h-fix1
```

## What it submits

- **Image**: `mcr.microsoft.com/azureml/curated/slime-pytorch-2.9-cuda12.8:3`
- **Entrypoint**: `${{inputs.code_dataset}}/retail_slime_train.py`
  (async GRPO via SLIME on Ray)
- **Distribution**: Ray, head + worker, dashboard on 8265
- **GPU layout**: `--num_gpus 8 --num_nodes 4 --tensor_parallel 8
  --rollout_num_gpus 16` (so 2 rollout nodes + 2 actor nodes)
- **Inputs**:
  - `train_dataset` → `retail-train-data`
  - `code_dataset`  → `retail-code`
  - `sft_lora_dataset` → checkpoint to seed RFT from
- **Outputs**: `model_output`, `checkpoints`, `rollouts`, `ray_temp`, `hf_cache`
- **Env**: `SLIME_HF_MODEL_ID=Qwen/Qwen3-32B`, `TAUBENCH_DOMAIN=retail`, etc.
- **Identity**: `fdp-training-pilot-umi` (UMI)

## When to bump

- Re-upload `retail-train-data` / `retail-code` → bump version strings in
  `DATASETS` at the top of the script.
- New SFT checkpoint to seed RFT from → update `sft_lora_dataset` URI.
- New container → update `ENV_IMAGE`.

Auth: `az login` first; the script uses `az account get-access-token`.
