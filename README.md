<a name="start-building"></a>
<p align="center">
<img src="img/banner-build-26.png" alt="Microsoft Build 2026" width="1200"/>
</p>

# [Microsoft Build 2026](https://build.microsoft.com)

## 🔥 BRK232: Post-Training and Deploying Open Source Reasoning Models in Foundry

### Session Description

Open-source reasoning models are powerful out of the box, but production performance comes from closing the loop. This session shows how to use **Microsoft Foundry** to collect production traces, curate them into datasets, and post-train reasoning models with reinforcement learning using frameworks like **slime**, **verl**, and **TRL** — then redeploy the improved models without managing the underlying GPU infrastructure. We cover when RL drives real gains versus other techniques, and walk through the full pipeline live on stage.

This repo contains the end-to-end source code shown in the session: an **SFT recipe** and an **async GRPO (RFT) recipe** for Qwen3 models, a multi-turn tool-use environment with a graded reward, and the helpers used to submit, monitor, and inspect training jobs on Foundry.

▶️ [Watch the session on the Microsoft Build 2026 site](https://build.microsoft.com/en-US/sessions/BRK232)

### 🚀 Getting started

This session demonstrates a real Foundry training run on multi-node A100 / H100 clusters. To reproduce it end to end you will need access to a Foundry project with attached GPU compute. Read through the [What's in this repo](#-whats-in-this-repo) section first to decide which recipe to start with.

> ⚠️ **Cloud costs apply.** Multi-node GPU training on Microsoft Foundry is expensive. When adapting the recipes, start with a small `--num_rollout` (RFT) or short SFT pass so you can validate end-to-end before scaling out.

#### Prerequisites

- A **[Microsoft Foundry](https://learn.microsoft.com/azure/ai-foundry/)** project. Copy the project endpoint URL from the project's **Overview** page — it has the format `https://<account>.services.ai.azure.com/api/projects/<project>`.
- A **GPU compute cluster** attached to the project. The recipes default to **4 nodes** of NVIDIA H100 (`ND96r_H100_v5`) or A100 (`ND96amsr_A100_v4`) capacity in Microsoft Foundry.
- A **user-assigned managed identity (UAI)** the training container will run as, plus the **storage connection name** in your Foundry workspace that the project MSI can write to.
- **Python 3.11 or newer** (the prerelease `azure-ai-projects` SDK uses `enum.StrEnum`).
- **[Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)** signed in to the tenant that owns your Foundry project (`az login`).

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

Make sure the active subscription owns your Foundry project: `az account show`.

#### 4. Stage 1 — Run the SFT recipe (Qwen3-32B)

Open [src/post-training-sft-recipe/zava_sft_submit.ipynb](src/post-training-sft-recipe/zava_sft_submit.ipynb) in Jupyter or Visual Studio Code, then in the first cell:

```python
from slime_sft_setup import setup_env, show_submit_params, submit_job, tail_rollouts

setup_env(project="<your-foundry-project-name>")
show_submit_params(cluster="h100")   # or cluster="a100"
submit_job(cluster="h100")
```

This submits a 4-node SLIME-on-Ray async-GRPO job that supervised-fine-tunes **Qwen3-32B** on the `zava` τ-bench-style domain. Use `tail_rollouts(...)` to stream rollout previews back to the notebook while the job runs. The recipe lives in [src/post-training-sft-recipe/recipe/submit_sft.py](src/post-training-sft-recipe/recipe/submit_sft.py); see its [README](src/post-training-sft-recipe/recipe/README.md) for what to bump when you re-upload data or change containers.

#### 5. Stage 2 — Run the RFT recipe (GRPO on Qwen3-14B, warm-started from the SFT LoRA)

Open [src/retail_rft_submit_styled.ipynb](src/retail_rft_submit_styled.ipynb) and run:

```python
from slime_rl_setup import setup_env, submit_job, stream_logs

setup_env(
    project_endpoint="https://<account>.services.ai.azure.com/api/projects/<project>",
    storage_connection_name="<your-storage-connection>",
    managed_identity_uai="/subscriptions/<sub>/resourcegroups/<rg>/providers/microsoft.managedidentity/userassignedidentities/<uai>",
    managed_identity_client_id="<uai-client-id>",
)
submit_job(cluster_name="<your-gpu-cluster>")
stream_logs()
```

This submits the **Retail post-purchase resolution** GRPO run — a multi-turn tool-use task with deterministic tools and an 8-component weighted grader (see [retail_grader_rft_tools_v3.py](src/post-training-recipe/demo-artifacts/code/retail_grader_rft_tools_v3.py)). The default `sft_lora_uri` in [submit_job.py](src/post-training-recipe/submit_job.py) points at the demo's SFT checkpoint; override it via the `sft_lora_uri` argument on `submit_job()` to chain in your own Stage-1 output.

> 🔧 The defaults baked into `submit_job.py` and `submit_sft.py` reference the **internal Foundry training pilot** project that hosted the on-stage demo. Override every `project_endpoint`, `managed_identity_*`, `storage_connection_name`, dataset URI, and `compute_cluster` value to point at your own project before submitting.

### 📦 What's in this repo

| Path | What it is |
|:-----|:-----------|
| [src/post-training-sft-recipe/](src/post-training-sft-recipe/) | **Stage 1 — SFT.** Notebook + recipe that submits a Qwen3-32B SLIME async-GRPO supervised fine-tune on 4×H100 or 4×A100. |
| [src/post-training-recipe/](src/post-training-recipe/) | **Stage 2 — RFT.** Notebook + recipe that submits a Qwen3-14B GRPO run on the Retail post-purchase task, warm-started from the SFT LoRA. |
| [src/post-training-recipe/demo-artifacts/code/](src/post-training-recipe/demo-artifacts/code/) | The custom **environment, tools, grader, and reward** for the multi-turn Retail task — the most reusable pieces if you're building your own RFT task. |
| [src/post-training-recipe/demo-artifacts/data/](src/post-training-recipe/demo-artifacts/data/) | Paraphrased customer scenarios (train / val JSONL) with expected actions, amounts, and tool sequences used by the grader. |
| [src/post-training-sft-recipe/reports/](src/post-training-sft-recipe/reports/) | Rollout-extraction utilities used to inspect what the model is producing during training. |

### 🧠 Learning Outcomes

By the end of this session, you will be able to:

- Use Microsoft Foundry as the control plane for collecting production traces, curating training datasets, and launching distributed post-training jobs.
- Decide **when reinforcement learning post-training is worth it** versus prompting, distillation, or plain SFT.
- Run an end-to-end **SFT → async GRPO** pipeline on open-source reasoning models (Qwen3-14B / Qwen3-32B) with the SLIME framework on Ray.
- Design a multi-turn tool-use environment and an **auditable, weighted reward function** that produces a stable learning signal.
- Redeploy the improved model back into Foundry without managing the underlying GPU infrastructure.

### 💬 Keep Learning with Copilot

Try these prompts in GitHub Copilot Chat (`Ctrl+Alt+I` on Windows/Linux, `Cmd+Shift+I` on macOS). Connect the [Microsoft Learn MCP Server](#-microsoft-learn-mcp-server) first so answers are grounded in the latest official documentation.

1. Understand the technique

```
Using the Microsoft Learn MCP Server, explain how Reinforcement Fine-Tuning (RFT) in Microsoft Foundry differs from SFT and DPO, and when I should pick each one.
```

2. Inspect the reward design

```
Open src/post-training-recipe/demo-artifacts/code/retail_grader_rft_tools_v3.py and explain how each of the 8 score components contributes to the final reward. Suggest one change that would make the grader more robust to format drift.
```

3. Adapt the recipe to a new task

```
I want to use the SLIME async-GRPO recipe in src/post-training-recipe/ to train Qwen3-14B on my own multi-turn tool-use dataset. Walk me through everything I'd need to change in submit_job.py, the env, the tools, and the grader.
```

### 💻 Technologies Used

1. **[Microsoft Foundry](https://learn.microsoft.com/azure/ai-foundry/)** — control plane for training jobs, datasets, managed identity, and model deployment.
1. **[Microsoft Foundry fine-tuning (SFT / DPO / RFT)](https://learn.microsoft.com/azure/ai-foundry/concepts/fine-tuning-overview)** — managed post-training of open-source and frontier models.
1. **[SLIME](https://github.com/THUDM/slime)** + **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** + **[Ray](https://www.ray.io/)** — async GRPO training stack used by both recipes.
1. **[SGLang](https://github.com/sgl-project/sglang)** — high-throughput rollout engine that pairs with the SLIME actor.
1. **[Qwen3-14B / Qwen3-32B](https://huggingface.co/Qwen)** — the open-source reasoning base models being post-trained.
1. **[azure-ai-projects (Python SDK)](https://learn.microsoft.com/python/api/overview/azure/ai-projects-readme)** + **[Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)** — job submission and authentication.

### 📚 Resources and Next Steps

| Resource | Description |
|:---------|:------------|
| [BRK232 session page](https://build.microsoft.com/en-US/sessions/BRK232) | Recording, abstract, and speaker info |
| [DEM321 — companion demo](https://build.microsoft.com/en-US/sessions/DEM321) | Shorter walkthrough of the same scenario shown in the demo theater |
| [Microsoft Foundry in Discord](https://aka.ms/build/foundrydiscord) | Discuss Foundry with the product team and community |
| [Microsoft Foundry fine-tuning concepts](https://learn.microsoft.com/azure/ai-foundry/concepts/fine-tuning-overview) | When to use SFT, DPO, and RFT |
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
