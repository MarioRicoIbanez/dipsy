# -*- coding: utf-8 -*-

import os 

os.system("git clone https://github.com/tloen/alpaca-lora.git")
os.system("pip install -q datasets loralib sentencepiece")
os.system("pip uninstall -y transformers")
os.system("pip install -q git+https://github.com/zphang/transformers@c3dc391")
os.system("pip install -q git+https://github.com/huggingface/peft.git")
os.system("pip install bitsandbytes")
os.system("pip install huggingface_hub")


import torch
from datasets import load_dataset
from transformers import LLaMATokenizer, AutoTokenizer, AutoConfig, LLaMAForCausalLM, GenerationConfig
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
import bitsandbytes as bnb
import transformers
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf", add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

data = load_dataset("json", data_files="alpaca-lora/alpaca_data.json")

def generate_prompt(data_point):
    if data_point["instruction"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""

data = data.map(lambda data_point: {"prompt": tokenizer(generate_prompt(data_point))})

MICRO_BATCH_SIZE = 4  
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 2  
LEARNING_RATE = 2e-5  
CUTOFF_LEN = 256  
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

model = LLaMAForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)
tokenizer = LLaMATokenizer.from_pretrained(
    "decapoda-research/llama-7b-hf", add_eos_token=True
)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  
data = load_dataset("json", data_files="alpaca-lora/alpaca_data.json")

def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
{data_point["output"]}"""

data = data.shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=1,
        output_dir="lora-alpaca",
        save_total_limit=3,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

model.save_pretrained("lora-alpaca")

import os
from huggingface_hub import HfApi

# Assumes your Hugging Face API token is stored in an environment variable
token = "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"

model.push_to_hub("RikoteMaster/alpaca7B-lora", use_auth_token=token)


tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")

model = LLaMAForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)

PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Tell me something about alpacas.
### Response:"""

inputs = tokenizer(
    PROMPT,
    return_tensors="pt",
)
input_ids = inputs["input_ids"].to(device)

generation_config = GenerationConfig(
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.15,
)
print("Generating...")
generation_output = model.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    return_dict_in_generate=True,
    output_scores=True,
    max_new_tokens=128,
)
for s in generation_output.sequences:
    print(tokenizer.decode(s))

PROMPT ='''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Write an ode to why do Alpacas make the best pets?

### Response:
'''

inputs = tokenizer(
    PROMPT,
    return_tensors="pt",
)
input_ids = inputs["input_ids"].to(device)

print("Generating...")
generation_output = model.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    return_dict_in_generate=True,
    output_scores=True,
    max_new_tokens=128,
)
for s in generation_output.sequences:
    print(tokenizer.decode(s))

def is_prime(number):  # Checks if a given integer is a prime number.
    for i in range (2, int(math.ceil(math.sqrt(number)))+1):
        if number % i == 0:
            return False
    return True

print(is_prime(23))
