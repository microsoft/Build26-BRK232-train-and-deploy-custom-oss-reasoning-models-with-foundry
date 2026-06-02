<a name="start-building"></a>
<p align="center">
<img src="img/banner-build-26.png" alt="Microsoft Build 2026" width="1200"/>
</p>

# [Microsoft Build 2026](https://build.microsoft.com)

## 🔥 BRK232: Post-Training and Deploying Open Source Reasoning Models in Foundry

### Session Description

Open-source reasoning models are powerful out of the box, but production performance comes from closing the loop. This session shows how to use **Microsoft Foundry** to collect production traces, curate them into datasets, and post-train reasoning models with reinforcement learning using **SLIME** on **Ray** — then redeploy the improved models without managing the underlying GPU infrastructure. We cover when RL post-training drives real gains versus prompting or plain SFT, and walk through the full pipeline live on stage.

This repo contains the complete source code shown in the session: a **Supervised Fine-Tuning (SFT)** recipe for Qwen3-32B, an **async GRPO (RFT)** recipe for Qwen3-14B warm-started from the SFT checkpoint, a multi-turn tool-use Retail environment with a deterministic 8-component grader, a live Streamlit rollout dashboard, and all the helpers used to submit, monitor, and inspect training jobs on Foundry.

▶️ [Watch the session on the Microsoft Build 2026 site](https://build.microsoft.com/en-US/sessions/BRK232)

### 🚀 Getting Started

This session demonstrates a real Foundry training run on multi-node A100 / H100 clusters. To reproduce it end-to-end you will need access to a Foundry project with attached GPU compute. Read through [What's in this repo](#-whats-in-this-repo) first to decide which recipe to start with.

> ⚠️ **Cloud costs apply.** Multi-node GPU training on Microsoft Foundry is expensive. Start with a small `--num_rollout` (RFT) or a short SFT pass to validate end-to-end before scaling out.

#### Prerequisites

- A **[Microsoft Foundry](https://learn.microsoft.com/azure/ai-foundry/)** project. Copy the project endpoint URL from the project's **Overview** page — format: `https://<account>.services.ai.azure.com/api/projects/<project>`.
- A **GPU compute cluster** attached to the project. The recipes default to **4 nodes** of NVIDIA H100 (`ND96r_H100_v5`) or A100 (`ND96amsr_A100_v4`) capacity.
- A **user-assigned managed identity (UAI)** the training container will run as, plus the **storage connection name** in your Foundry workspace that the project MSI can write to.
- **Python 3.11 or newer** (the prerelease `azure-ai-projects` SDK uses `enum.StrEnum`).
- **[Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)** signed in to the tenant that owns your Foundry project (`az login`).

> 🔬 **Private Preview.** The Foundry Custom Code training require access to the private preview program. Contact your Microsoft account team if you don't have access yet.

#### 1. Clone the repo

```bash
git clone https://github.com/microsoft/Build26-BRK232-train-and-deploy-custom-oss-reasoning-models-with-foundry.git
cd Build26-BRK232-train-and-deploy-custom-oss-reasoning-models-with-foundry
```

#### 2. Install the Python dependencies

```bash
pip install --pre -r src/requirements.txt \
  --extra-index-url https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple
```

This installs the prerelease `azure-ai-projects` SDK and `azure-identity` used by both notebooks to submit jobs.

#### 3. Sign in to Azure

```bash
az login
```

Verify the active subscription owns your Foundry project: `az account show`.

#### 4. Stage 1 — Run the SFT recipe (Qwen3-32B)

Open [src/post-training-sft-recipe/retail_sft_submit.ipynb](src/post-training-sft-recipe/retail_sft_submit.ipynb) in Jupyter or VS Code, then in the first cell:

```python
from slime_sft_setup import setup_env, show_submit_params, submit_job, tail_rollouts

setup_env(project="<your-foundry-project-name>")
show_submit_params(cluster="h100")   # or cluster="a100"
submit_job(cluster="h100")
```

This submits a 4-node SLIME-on-Ray supervised fine-tuning job on **Qwen3-32B** against the Retail domain dataset. Use `tail_rollouts(...)` to stream rollout previews back to the notebook while the job runs. The recipe payload is in [src/post-training-sft-recipe/recipe/submit_sft.py](src/post-training-sft-recipe/recipe/submit_sft.py) — see its [README](src/post-training-sft-recipe/recipe/README.md) for what to bump when re-uploading data or changing containers.

#### 5. Stage 2 — Run the RFT recipe (GRPO on Qwen3-14B, warm-started from the SFT LoRA)

Open [src/Retail_Customer_Agent_Post_Training.ipynb](src/Retail_Customer_Agent_Post_Training.ipynb) and run:

```python
from slime_rl_setup import setup_env, submit_job, job_status

setup_env(
    project_endpoint="https://<account>.services.ai.azure.com/api/projects/<project>",
    storage_connection_name="<your-storage-connection>",
    managed_identity_uai="/subscriptions/<sub>/resourcegroups/<rg>/providers/microsoft.managedidentity/userassignedidentities/<uai>",
    managed_identity_client_id="<uai-client-id>",
)
submit_job(cluster_name="<your-gpu-cluster>")
job_status()   # polls job state; opens Foundry portal link for logs
```

This submits the **Retail post-purchase resolution** GRPO run — a multi-turn tool-use task with deterministic tools and an 8-component weighted grader (see [`retail_grader_rft_tools_v3.py`](src/post-training-recipe/demo-artifacts/code/retail_grader_rft_tools_v3.py)). The default `sft_lora_dataset_id` in [`submit_job.py`](src/post-training-recipe/submit_job.py) points at the demo's SFT checkpoint; override it via the `sft_lora_uri` argument on `submit_job()` to chain in your own Stage 1 output.

> 🔧 The defaults baked into `submit_job.py` and `submit_sft.py` reference the **internal Foundry training pilot** project used for the on-stage demo. Override every `project_endpoint`, `managed_identity_*`, `storage_connection_name`, dataset URI, and `compute_cluster` value to point at your own project before submitting.

#### 6. Stage 3 — Run using the Foundry Low-Level APIs (Qwen3-32B, maximum control)

> 🔬 **Private Preview.** The Foundry Finetuning Low-Level APIs require access to the private preview program. Contact your Microsoft account team if you don't have access yet.

Open [src/Retail_Customer_Agent_Training_API.ipynb](src/Retail_Customer_Agent_Training_API.ipynb) for a fully custom GRPO training loop that runs **from your local machine** while all GPU compute happens on Azure. This approach gives you direct control over batching, rollout strategies, and curriculum scheduling — things not yet exposed through the high-level SDK.

The loop uses three primitives from the Foundry Low-Level API:

| Primitive | What it does |
|:----------|:------------|
| `client.create_session(...)` | Provisions a LoRA adapter on the Azure GPU cluster |
| `client.sample(...)` | Runs multi-turn rollouts (the model calls tools just like in production) |
| `client.train(...)` | Applies gradient updates server-side; you never download full model weights |

Set your credentials, then run the notebook cells top-to-bottom:

```bash
export AZURE_AI_API_KEY="<key>"   # or use az login
export PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
```

Results from the on-stage demo: **Qwen3-32B improved from 58.1% → 86.9%** retail_quality (+28.8pp), outperforming o4-mini (82.3%) and GPT-4.1-mini SFT (71.0%).

### 📦 What's in this repo

| Path | What it is |
|:-----|:-----------|
| [src/post-training-sft-recipe/retail_sft_submit.ipynb](src/post-training-sft-recipe/retail_sft_submit.ipynb) | **Stage 1 notebook.** Submits a Qwen3-32B SLIME supervised fine-tune on 4×H100 or 4×A100. |
| [src/post-training-sft-recipe/slime_sft_setup.py](src/post-training-sft-recipe/slime_sft_setup.py) | Helper for the SFT notebook — wraps `setup_env`, `show_submit_params`, `submit_job`, and `tail_rollouts`. |
| [src/post-training-sft-recipe/recipe/](src/post-training-sft-recipe/recipe/) | SFT recipe script (`submit_sft.py`) and its README — the exact job payload sent to Foundry. |
| [src/post-training-sft-recipe/demo-artifacts/code/sft_retail.py](src/post-training-sft-recipe/demo-artifacts/code/sft_retail.py) | HF TRL-based SFT training script used inside the Foundry container for prompt-masked fine-tuning. |
| [src/post-training-sft-recipe/demo-artifacts/data/](src/post-training-sft-recipe/demo-artifacts/data/) | SFT training data — `retail_train_sft.jsonl` and `retail_val_sft.jsonl`. |
| [src/post-training-sft-recipe/reports/extract_rollouts.py](src/post-training-sft-recipe/reports/extract_rollouts.py) | Utility to extract and inspect rollout outputs from a running or completed SFT job. |
| [src/Retail_Customer_Agent_Post_Training.ipynb](src/Retail_Customer_Agent_Post_Training.ipynb) | **Stage 2 notebook.** Submits a Qwen3-14B GRPO run warm-started from the Stage 1 SFT LoRA. |
| [src/slime_rl_setup.py](src/slime_rl_setup.py) | Helper for the RFT notebook — wraps `setup_env`, `submit_job`, and `job_status`. |
| [src/post-training-recipe/submit_job.py](src/post-training-recipe/submit_job.py) | Builds the full Foundry `CommandJob` body and submits the Retail GRPO run. |
| [src/post-training-recipe/helpers.py](src/post-training-recipe/helpers.py) | SDK-based helpers for dataset upload, GPU layout, job body construction, and submission. |
| [src/post-training-recipe/demo-artifacts/code/retail_env.py](src/post-training-recipe/demo-artifacts/code/retail_env.py) | In-memory Retail environment — dispatches deterministic tools and tracks episode state. |
| [src/post-training-recipe/demo-artifacts/code/retail_tools.py](src/post-training-recipe/demo-artifacts/code/retail_tools.py) | Deterministic tool implementations (`get_order_details`, `check_resolution_policy`, etc.). |
| [src/post-training-recipe/demo-artifacts/code/retail_grader_rft_tools_v3.py](src/post-training-recipe/demo-artifacts/code/retail_grader_rft_tools_v3.py) | 8-component weighted grader (verb, item, reason, format, amount, tool coverage, workflow, integrity). |
| [src/post-training-recipe/demo-artifacts/code/retail_reward.py](src/post-training-recipe/demo-artifacts/code/retail_reward.py) | Custom reward module — calls the grader and shapes the scalar reward signal for GRPO. |
| [src/post-training-recipe/demo-artifacts/code/retail_slime_train.py](src/post-training-recipe/demo-artifacts/code/retail_slime_train.py) | SLIME training entrypoint — runs inside the Foundry container to launch Ray + GRPO training. |
| [src/post-training-recipe/demo-artifacts/code/dashboard.py](src/post-training-recipe/demo-artifacts/code/dashboard.py) | Streamlit rollout browser — exposed as a live Foundry job service on port 8501 during training. |
| [src/post-training-recipe/demo-artifacts/data/](src/post-training-recipe/demo-artifacts/data/) | RFT training data — `retail_train.jsonl` and `retail_val.jsonl` with expected actions and tool sequences. |
| [src/post-training-experimentation/](src/post-training-experimentation/) | Local grader evaluation tools — `grader_demo.py`, `debug_grader.py`, `grader_eval_helpers.py`, and sample data for offline testing before submitting a run. |
| [src/Retail_Customer_Agent_Grader_Test_Bed.ipynb](src/Retail_Customer_Agent_Grader_Test_Bed.ipynb) | Interactive notebook for testing and iterating on grader logic locally. |
| [src/Retail_Customer_Agent_Training_API.ipynb](src/Retail_Customer_Agent_Training_API.ipynb) | **Stage 3 — Low-Level APIs notebook.** Uses the Foundry Finetuning Low-Level APIs (private preview) to run a fully custom GRPO loop for Qwen3-32B, including multi-turn agent rollouts, live metrics, and model deployment. Achieves **58.1% → 86.9%** retail_quality (+28.8pp). |

### 🧠 Learning Outcomes

By the end of this session, you will be able to:

- Use **Microsoft Foundry** as the control plane for curating training datasets and launching distributed post-training jobs.
- Decide **when reinforcement learning post-training is worth it** versus prompting, distillation, or plain SFT.
- Run an end-to-end **SFT → async GRPO** pipeline on open-source reasoning models (Qwen3-14B / Qwen3-32B) using the SLIME framework on Ray.
- Use the **Foundry Finetuning Low-Level APIs** to build a fully custom GRPO training loop that runs locally while GPU compute happens on Azure.
- Design a multi-turn tool-use environment and an **auditable, weighted reward function** that produces a stable learning signal.
- Redeploy the improved model back into Foundry without managing the underlying GPU infrastructure.

### 💬 Keep Learning with Copilot

Try these prompts in GitHub Copilot Chat (`Ctrl+Alt+I` on Windows/Linux, `Cmd+Shift+I` on macOS). Connect the [Microsoft Learn MCP Server](#-microsoft-learn-mcp-server) first so answers are grounded in the latest official documentation.

1. Understand the technique

```
Using the Microsoft Learn MCP Server, explain how Reinforcement Fine-Tuning (RFT) in Microsoft Foundry differs from SFT, and when I should pick each one.
```

2. Inspect the reward design

```
Open src/post-training-recipe/demo-artifacts/code/retail_grader_rft_tools_v3.py and explain how each of the 8 score components contributes to the final reward. Suggest one change that would make the grader more robust to format drift.
```

3. Adapt the recipe to a new task

```
I want to use the SLIME async-GRPO recipe in src/post-training-recipe/ to train Qwen3-14B on my own multi-turn tool-use dataset. Walk me through everything I'd need to change in submit_job.py, retail_env.py, retail_tools.py, and the grader.
```

### 💻 Technologies Used

1. **[Microsoft Foundry](https://learn.microsoft.com/azure/ai-foundry/)** — control plane for training jobs, datasets, managed identity, and model deployment.
1. **[Microsoft Foundry fine-tuning (SFT / RFT)](https://learn.microsoft.com/azure/ai-foundry/concepts/fine-tuning-overview)** — managed post-training of open-source models.
1. **[SLIME](https://github.com/THUDM/slime)** + **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** + **[Ray](https://www.ray.io/)** — async GRPO training stack used by both recipes.
1. **[SGLang](https://github.com/sgl-project/sglang)** — high-throughput rollout engine that pairs with the SLIME actor.
1. **[TRL (Hugging Face)](https://huggingface.co/docs/trl)** — used by the SFT recipe for prompt-masked supervised fine-tuning.
1. **[Streamlit](https://streamlit.io/)** — live rollout browser dashboard (`dashboard.py`) served as a Foundry job service during training.
1. **[Qwen3-14B / Qwen3-32B](https://huggingface.co/Qwen)** — the open-source reasoning base models being post-trained.
1. **[azure-ai-projects (Python SDK)](https://learn.microsoft.com/python/api/overview/azure/ai-projects-readme)** + **[Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)** — job submission and authentication.

### 📚 Resources and Next Steps

| Resource | Description |
|:---------|:------------|
| [BRK232 session page](https://build.microsoft.com/en-US/sessions/BRK232) | Recording, abstract, and speaker info |
| [DEM321 — companion demo](https://build.microsoft.com/en-US/sessions/DEM321) | Shorter walkthrough of the same scenario shown in the demo theater |
| [Microsoft Foundry in Discord](https://aka.ms/build/foundrydiscord) | Discuss Foundry with the product team and community |
| [Microsoft Foundry fine-tuning concepts](https://learn.microsoft.com/azure/ai-foundry/concepts/fine-tuning-overview) | When to use SFT and RFT |
| [SLIME framework](https://github.com/THUDM/slime) | Async RL framework used by both recipes in this repo |
| [https://aka.ms/build26-next-steps](https://aka.ms/build26-next-steps) | Explore lab and session repos to further your learning from Microsoft Build |


### 🌟 Microsoft Learn MCP Server

The Microsoft Learn MCP Server gives your AI agent direct access to Microsoft's official documentation — grounded, up-to-date answers about the products and services covered in this session.

**VS Code** — One click installation: 

[![Install in VS Code](https://img.shields.io/badge/VS_Code-Install_Microsoft_Learn_MCP-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](https://vscode.dev/redirect/mcp/install?name=microsoft-learn&config=%7B%22type%22%3A%22http%22%2C%22url%22%3A%22https%3A%2F%2Flearn.microsoft.com%2Fapi%2Fmcp%22%7D)


**GitHub Copilot CLI** — Run this to install the Learn MCP Server as a plugin:
```
/plugin install microsoftdocs/mcp
```

For more info, other clients, and to post questions, visit the [Learn MCP Server repo](https://aka.ms/learnmcp).

## Content Owners

<table>
<tr>
    <td align="center"><a href="https://github.com/vijayaski">
        <img src="https://github.com/vijayaski.png" width="100px;" alt="Vijay Aski"/><br />
        <sub><b>Vijay Aski</b></sub></a><br />
        <sub>Senior Partner Director, Microsoft</sub><br />
        <a href="https://build.microsoft.com/en-US/sessions/BRK232" title="talk">📢</a>
    </td>
    <td align="center">
        <img src="https://avatars.githubusercontent.com/u/9919?s=100&v=4" width="100px;" alt="Manoj Bableshwar"/><br />
        <sub><b>Manoj Bableshwar</b></sub><br />
        <sub>Microsoft</sub><br />
        <a href="https://build.microsoft.com/en-US/sessions/BRK232" title="talk">📢</a>
    </td>
    <td align="center"><a href="https://www.linkedin.com/in/clauren">
        <img src="https://avatars.githubusercontent.com/u/9919?s=100&v=4" width="100px;" alt="Chris Lauren"/><br />
        <sub><b>Chris Lauren</b></sub></a><br />
        <sub>Partner Group Product Manager, Core AI, Microsoft</sub><br />
        <a href="https://build.microsoft.com/en-US/sessions/BRK232" title="talk">📢</a>
    </td>
</tr></table>

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

