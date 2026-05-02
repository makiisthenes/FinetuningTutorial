# FoodExtract — Fine-tuning a Small LLM with SFT

A hands-on, end-to-end walkthrough of **Supervised Fine-Tuning (SFT)** on a small open-source LLM. The notebook in this repo takes Google's [`gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it) (a 270M parameter instruction-tuned model) and teaches it one specific job: extract food and drink items from a piece of text and return them in a compact, structured format.

The result is a tiny, task-specific model that runs locally — no API calls, no data leaving your machine.

## Live demo

- **Try it:** [Hugging Face Space (Gradio)](https://huggingface.co/spaces/makiisthebes/gemma-3-270M-FoodExtract-demo)
- **Model:** [`makiisthebes/gemma-3-270M-Instruct-FoodExtract`](https://huggingface.co/makiisthebes/gemma-3-270M-Instruct-FoodExtract)

### Example

**Input**

```
For breakfast I had eggs, bacon and toast and a glass of orange juice
```

**Output**

```
food_or_drink: 1
tags: fi, di
foods: eggs, bacon, toast, orange juice
drinks: orange juice
```

The output uses a [TOON](https://github.com/toon-format/toon-python)-style format — chosen over JSON because it's significantly cheaper in tokens, which matters when training and serving a 270M model.

## What this repo covers

The notebook walks through the full SFT pipeline, end to end:

1. Loading a base model and tokenizer with `AutoModelForCausalLM` / `AutoTokenizer`
2. Why **chat templates** matter (and what happens without one)
3. Building a structured training dataset and converting it to the model's chat format
4. Train / validation / test splits
5. Configuring and running `SFTTrainer` from [TRL](https://huggingface.co/docs/trl)
6. Plotting loss curves and reading them (overfitting in LMs, when it's actually fine)
7. Qualitative evaluation on the held-out test set
8. Pushing the fine-tuned model to the Hugging Face Hub
9. Wrapping it in a Gradio demo and deploying to Hugging Face Spaces

## Repo structure

```
.
├── finetuning_setup.ipynb     # Main tutorial notebook (start here)
├── food-extract-demo/         # Gradio demo deployed to HF Spaces
│   ├── app.py                 # Gradio interface
│   ├── upload_to_hf.py        # Pushes the demo folder to a HF Space
│   ├── requirements.txt
│   └── README.md
├── images/                    # Screenshots referenced in the notebook
├── pyproject.toml
└── README.md
```

> **Note:** The trained checkpoints (`checkpoint_models/`) are *not* tracked in git — they're gigabytes each. The final model lives on the [Hugging Face Hub](https://huggingface.co/makiisthebes/gemma-3-270M-Instruct-FoodExtract); regenerate intermediate checkpoints by re-running the notebook.

## Quickstart

### 1. Run the Gradio demo locally

```bash
cd food-extract-demo
pip install -r requirements.txt
python app.py
```

This downloads the fine-tuned model from the Hub on first run and serves the demo at `http://127.0.0.1:7860`.

### 2. Re-run the fine-tuning notebook

Requires Python 3.11+ and a CUDA-capable GPU for reasonable training speed.

```bash
pip install transformers trl datasets torch jupyter
jupyter lab finetuning_setup.ipynb
```

Set `HF_TOKEN` in a `.env` file (or via `huggingface-cli login`) if you want to push your own fine-tuned model back to the Hub.

### 3. Re-deploy the demo to your own Space

```bash
export HF_TOKEN=hf_...
cd food-extract-demo
python upload_to_hf.py
```

Edit `HF_USERNAME` and `HF_SPACE_NAME` in `upload_to_hf.py` first.

## Key takeaways

- **Small models are enough for narrow tasks.** A 270M model fine-tuned on a few thousand examples can reliably do structured extraction — no need for a 70B model or a paid API.
- **Chat templates are non-negotiable.** Without `apply_chat_template`, the base model just autocompletes instead of answering — a common gotcha for first-time fine-tuners.
- **Output format matters for token cost.** TOON over JSON cuts training and inference tokens significantly without losing structure.
- **A bit of overfitting is okay** when the task is narrow and the desired output structure is highly repetitive.

## Credits

Based on the fine-tuning live streams by [@TrelisResearch](https://www.youtube.com/@TrelisResearch):

- [Part 1 — Fine-tuning a small LLM with SFT](https://www.youtube.com/live/M32fJdG3D_A)
- [Part 2 — Publishing, demoing and evaluating](https://www.youtube.com/live/zt8t-xT26rA)

Built with: [transformers](https://github.com/huggingface/transformers), [TRL](https://github.com/huggingface/trl), [datasets](https://github.com/huggingface/datasets), [Gradio](https://github.com/gradio-app/gradio), and [TOON](https://github.com/toon-format/toon-python).

## License

Apache 2.0 — same as the base model.
