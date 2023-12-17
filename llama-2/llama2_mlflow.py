import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


import mlflow

#create mlflow experiment
mlflow.set_experiment("llama2")





# The model that you want to train from the Hugging Face hub
model_name = "meta-llama/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "RikoteMaster/emotion_recog_sample"

# Fine-tuned model name
new_model = "llama-2-7b-sentiment-analyzer"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results_selected"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True

# Batch size per GPU for training
per_device_train_batch_size = 16

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 3e-6

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 25

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}


#log params as different values

run_name = f"Entrenando {model_name} con {dataset_name}"
mlflow.set_tracking_uri("/workspace/NASFolder/mlruns")
mlflow.set_experiment("LLMs")
mlflow.start_run(run_name=run_name)

mlflow.log_param("model_name", model_name)
mlflow.log_param("dataset_name", dataset_name)
mlflow.log_param("new_model", new_model)
mlflow.log_param("lora_r", lora_r)
mlflow.log_param("lora_alpha", lora_alpha)
mlflow.log_param("lora_dropout", lora_dropout)
mlflow.log_param("use_4bit", use_4bit)
mlflow.log_param("bnb_4bit_compute_dtype", bnb_4bit_compute_dtype)
mlflow.log_param("bnb_4bit_quant_type", bnb_4bit_quant_type)
mlflow.log_param("use_nested_quant", use_nested_quant)
mlflow.log_param("output_dir", output_dir)
mlflow.log_param("num_train_epochs", num_train_epochs)
mlflow.log_param("fp16", fp16)
mlflow.log_param("bf16", bf16)
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("weight_decay", weight_decay)
mlflow.log_param("optim", optim)
mlflow.log_param("lr_scheduler_type", lr_scheduler_type)
mlflow.log_param("max_steps", max_steps)
mlflow.log_param("warmup_ratio", warmup_ratio)
mlflow.log_param("group_by_length", group_by_length)
mlflow.log_param("save_steps", save_steps)
mlflow.log_param("logging_steps", logging_steps)
mlflow.log_param("max_seq_length", max_seq_length)
mlflow.log_param("packing", packing)
mlflow.log_param("device_map", device_map)

################################################################################
# Load dataset (you can process it here)
dataset = load_dataset(dataset_name, split="train")

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type
    )

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)




from datasets import load_dataset

ds = load_dataset("RikoteMaster/isear_augmented_sample")

texts = ds['test']['Text_processed']
labels = ds['test']['Emotion']


label_detection = []
wrong_detection = []
corrects = 0
exceptions = 0
for sentence, label in zip(texts, labels):
    text = f"""<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows: Anger, Joy, Sadnes, Guilt, Shame, fear or disgust Sentence: {sentence} [/INST] """
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=len(tokenizer(text)) + 200)
    result = pipe(text)
    try:
        detected = result[0]['generated_text'].split('[/INST]')[1].split()[0]
        if label != detected:
            wrong_detection.append(str(result) + " THE TRUE LABEL IS "+ label)
            label_detection.append(detected)
        else:
            corrects += 1
    except:
        
        wrong_detection.append(str(result) + " THE TRUE LABEL IS " + label)
        label_detection.append(detected)
        exceptions += 1 
        


print(corrects/len(texts))



mlflow.log_metric("accuracy", corrects/len(texts))
mlflow.log_metric("exceptions", exceptions)
mlflow.log_metric("corrects", corrects)
mlflow.log_metric("wrong", len(texts) - corrects)
mlflow.log_metric("total", len(texts))
mlflow.end_run()


