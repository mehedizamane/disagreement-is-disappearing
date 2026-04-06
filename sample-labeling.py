#!/usr/bin/env python

import pandas as pd
import torch
import re
from transformers import AutoTokenizer, TextStreamer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# -------------------------------
# Load the fine-tuned model
# -------------------------------
model_name = "/scratch/sz841/unsloth_deepseek_v16/model/"  # Adjust path if needed
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=False,  # Change to True if you trained in 4-bit
    device_map="auto"  # Auto-allocate GPU
)

# Enable faster inference
FastLanguageModel.for_inference(model)

# Load chat template
chat_template = get_chat_template(tokenizer, chat_template="llama-3.1")

# -------------------------------
# Load the Input CSV File
# -------------------------------
input_csv = "/scratch/sz841/unsloth_deepseek_v16/glenn/glenn_guest_utterances.csv"  # Change if needed
df = pd.read_csv(input_csv)

# -------------------------------
# Extract Label from Model Output
# -------------------------------
def extract_label(model_output):
    """Extracts only the classification label from the model output."""
    match = re.search(r"assistant\s*\n(.+)", model_output, re.DOTALL)
    return match.group(1).strip() if match else "UNKNOWN"

# -------------------------------
# Define Inference Function
# -------------------------------
# Function to prepare input for inference and generate label
def generate_label(speaker1, speaker2):
    instruction = (
        "You are classifying TV show dialogues (just one directional) between two speakers - host (speaker 1) and guest (speaker 2) into one of three labels: agreement, disagreement, or neutrality.\n\n"
        "IMPORTANT RULES:\n"
        "- ALWAYS classify as 'neutrality' if Speaker 1 asks a question and Speaker 2 directly answers that question.\n"
        "- ALWAYS classify as 'neutrality' if Speaker 1 introduces a video commentary and ends with the word 'watch'.\n"
        "- Otherwise, determine if the interaction represents 'agreement,' 'disagreement,' or 'neutrality.'\n\n"
        "Now classify this interaction:\n"
    )

    messages = [
        {"role": "user", "content": f"{instruction}\nSpeaker 1: {speaker1}\nSpeaker 2: {speaker2}"}
    ]



    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")  # Move to GPU

    with torch.no_grad():
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        output_ids = model.generate(
            input_ids=inputs,
            streamer=text_streamer,
            max_new_tokens=10,  # Adjust token limit as needed
            use_cache=True,
            temperature=0.1,  # Reduce randomness
            min_p=0.1,
            top_p=0.9
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return extract_label(output_text)

# -------------------------------
# Run Inference and Extract Labels
# -------------------------------
df["model_label"] = df.apply(lambda row: generate_label(row["speaker1"], row["speaker2"]), axis=1)

# -------------------------------
# Save Outputs to CSV
# -------------------------------
output_csv = "/scratch/sz841/unsloth_deepseek_v16/glenn/glenn_model_labels.csv"
df.to_csv(output_csv, index=False)

print(f"✅ Inference complete. Labels saved to {output_csv}")
