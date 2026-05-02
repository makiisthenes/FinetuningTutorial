import time
import torch
import gradio as gr

from typing import List, Tuple
from transformers import pipeline

# Initialise the pipeline once at module level so it is not reloaded on every call
pipe = pipeline(
    task="text-generation",
    model="makiisthebes/gemma-3-270M-Instruct-FoodExtract",
    device="cuda" if torch.cuda.is_available() else "cpu",
)


def _parse_model_output(model_output: List[dict]) -> str:
    start_model_string = "<start_of_turn>model"
    if len(model_output) != 1:
        raise ValueError(f"Unexpected model output format: {model_output}")
    raw_response = model_output[0].get("generated_text", "")
    parsed_response = raw_response[raw_response.index(start_model_string) + len(start_model_string):]
    return parsed_response


def extract_food(prompt: str) -> Tuple[str, str, float]:
    start_time = time.time()
    model_input = pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    raw_output = pipe(model_input)
    elapsed = round(time.time() - start_time, 4)
    generated_text = _parse_model_output(raw_output)
    return generated_text, raw_output[0]["generated_text"], elapsed


description = """
Extract food and drink items from text with a fine-tuned SLM.

* Input (str): Raw text strings or image captions (e.g. "A photo of a person's lunch with a tuna, cheese and capers melt sandwich")
* Output (str): Generated text with food/not_food classification as well as noun extraction.

For example:

Input: "For breakfast I had eggs, bacon and toast and a glass of orange juice"
Output:
food_or_drink: 1
tags: fi, di
foods: eggs, bacon, toast, orange juice
drinks: orange juice
"""

demo = gr.Interface(
    fn=extract_food,
    inputs=gr.TextArea(lines=4, label="Input Text"),
    outputs=[
        gr.TextArea(lines=4, label="Parsed Output"),
        gr.TextArea(lines=7, label="Raw Output"),
        gr.Number(label="Generation Time (s)"),
    ],
    title="FoodExtract with a Fine-tuned Gemma 3 270M model.",
    description=description,
    examples=[
        ["Hello world! This is my first fine-tuned llm."],
        ["a photo of a person's lunch with a tuna, cheese and capers melt sandwich"],
        ["A table of various indian foods including golden samosas with crispy edges."],
    ],
)

if __name__ == "__main__":
    demo.launch(share=False)
