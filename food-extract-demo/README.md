---
title: Food Extract
emoji: 🍗🥗🍹
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# 🍗🥗🍹 Food Extract

Small demo to showcase a fine-tuned SLM that extracts food and drink items from raw text or image captions.

Gemma 3 270M Instruct model fine-tuned using SFT (Supervised Fine-Tuning) on a synthetic dataset of food and non-food image captions.

Given an input string the model outputs a structured response:

```
food_or_drink: 1
tags: fi, di
foods: eggs, bacon, toast
drinks: orange juice
```

Model: [makiisthebes/gemma-3-270M-Instruct-FoodExtract](https://huggingface.co/makiisthebes/gemma-3-270M-Instruct-FoodExtract)

[Source code notebook](https://github.com/makiisthebes/FinetuningTutorial/blob/main/finetuning_setup.ipynb)
