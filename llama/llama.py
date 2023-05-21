# -*- coding: utf-8 -*-
"""
This script installs necessary libraries, defines functions for data preprocessing and model training,
and trains a sentiment analysis model using the Hugging Face's Transformer library.
"""

# Installing necessary libraries


# Importing necessary libraries
import transformers 
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM 
import os
import sys
from typing import List
from peft import (
  LoraConfig, 
  get_peft_model,
  get_peft_model_state_dict,
  prepare_model_for_int8_training,
)

import fire 
import torch
from datasets import load_dataset 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import seaborn as sns 
from pylab import rcParams 
import json
import re
import string
sns.set(rc={'figure.figsize': (8, 6)}) 
sns.set (rc={'figure.dpi':100})
sns.set(style='white', palette='muted', font_scale=1.2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Function to clean text
def clean_text(text):
    """
    This function takes a text string and performs the following:
    1. Converts the text to lowercase
    2. Removes URLs
    3. Removes punctuation
    4. Removes words containing numbers
    5. Removes multiple spaces
    Returns the cleaned text.
    """
    text = text.lower()
    text = re.sub('https:\/\/\S+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'[^ \w\.]', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(' +', ' ', text)
    return text

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    """
    This function loads and preprocesses data from a csv file, applies the clean_text function,
    and adds a new column 'Text_processed'
    """
    df = pd.read_csv(file_path, names=['Emotion', 'Text', 'DNTKNOW'], sep=',').drop(columns=['DNTKNOW']).dropna()
    df['Text_processed'] = df.Text.apply(clean_text)
    return df

# Load and preprocess data
url = 'https://raw.githubusercontent.com/PoorvaRane/Emotion-Detector/master/ISEAR.csv'
df = load_and_preprocess_data(url)
df.Emotion.value_counts().plot(kind="bar")

# Create dataset for training
dataset_data = [
    {
        "instruction": "Detect the sentiment.",
        "input": row_dict["Text_processed"],
        "output": row_dict["Emotion"]
    }
  for row_dict in df.to_dict(orient="records")
]
with open("isear.json", "w") as f:
    json.dump(dataset_data, f)

# Load pretrained model and tokenizer
BASE_MODEL = "decapoda-research/llama-7b-hf"
model = LlamaForCausalLM.from_pretrained(
      BASE_MODEL,
      load_in_8bit=True,
      torch_dtype=torch.float16,
      device_map="auto"
)
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

# Load dataset
data = load_dataset("json", data_files="isear.json")

# Define functions for tokenizing prompts
def generate_prompt(data_point):
    """
    This function generates a prompt for the model based on a data point.
    """
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
    ### Instruction:
    {data_point["instruction"]}
    ### Input:
    {data_point["input"]}
    ### Response:
    {data_point["output"]}"""

CUTOFF_LEN = 256

def tokenize(prompt, add_eos_token=True):
    """
    This function tokenizes a prompt.
    """
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
 
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    """
    This function generates a prompt for a data point and tokenizes it.
    """
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

# Split data into train and validation sets
train_val = data["train"].train_test_split(
    test_size = 0.2, shuffle=True, seed=42
)
train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
val_data = train_val["test"].map(generate_and_tokenize_prompt)

# Set up model for training
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 300
OUTPUT_DIR = "experiments"
model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

model.print_trainable_parameters()

# Set up training arguments
training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=False,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard"
)

# Create data collator and trainer
data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))
model = torch.compile(model)

# Train the model
trainer.train()
model.save_pretrained(OUTPUT_DIR)

# Authenticate with Hugging Face and push model to the hub

import huggingface_hub
huggingface_hub.login(token = "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")

model.push_to_hub("RikoteMaster/sentiment_analysys_isear", use_auth_token=True)

