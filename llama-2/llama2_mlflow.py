import re
import torch
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging
)
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
import mlflow


import random 
import numpy as np 

import os 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define a data class for your custom arguments
@dataclass
class CustomTrainingArguments:
    model_name: str = field(metadata={"help": "Model identifier from the Hugging Face Hub."})
    dataset_name: str = field(metadata={"help": "Dataset identifier from the Hugging Face Hub."})
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    explainning: bool = field(default=False, metadata={"help": "If the model is going to be trained with explanation or not."})

# Parse arguments from the command line
parser = HfArgumentParser(CustomTrainingArguments)
training_args, = parser.parse_args_into_dataclasses()

# Assign parsed arguments to variables
model_name = training_args.model_name
dataset_name = training_args.dataset_name
num_train_epochs = training_args.num_train_epochs
learning_rate = training_args.learning_rate

# Define the regular expression pattern
pattern = r'[^/]+$'

# Find the matches in the model_name and dataset_name strings
model_match = re.findall(pattern, model_name)
dataset_match = re.findall(pattern, dataset_name)

# Concatenate the matches with a dash to form the new_model name
new_model = model_match[0] + '-' + dataset_match[0]

mlflow.set_experiment(new_model)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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
output_dir = "/checkpoints"

# Number of training epochs

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True

# Batch size per GPU for training
per_device_train_batch_size = 32

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3


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
save_steps = 500

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
mlflow.set_tracking_uri("http://158.42.170.253:5000")
mlflow.set_experiment("LLMs_seed")
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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
trainer.model.push_to_hub(new_model, private=True)
tokenizer.push_to_hub(new_model, private=True)




from datasets import load_dataset

ds = load_dataset("RikoteMaster/isear_rauw")

texts = ds['test']['Text_processed']
labels = ds['test']['Emotion']




# Initialize counters and lists for tracking
corrects = 0
wrong_detection = []
label_detection = []
exceptions = 0
predicted_labels = [] 


for sentence, label in zip(texts, labels):
    if not training_args.explainning:
        text = f"""<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows: Anger, Joy, Sadnes, Guilt, Shame, fear or disgust Sentence: {sentence} [/INST]"""
    else:
        text = f"""<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows: Anger, Joy, Sadness, Guilt, Shame, Fear, or Disgust. Firstly, you have to express the explanation of why you think it's one emotion or another to make a pre-explanation. After that, you will predict the emotion expressed by the sentence. The format will be Explanation: and later Emotion: Sentence: {sentence} [/INST] """
    
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=len(tokenizer.tokenize(text)) + 5 if not training_args.explainning else len(tokenizer.tokenize(text)) + 70)
    result = pipe(text)
    print(result)
    try:
        pattern = r'\b(anger|joy|sadness|guilt|shame|fear|disgust)\b'

        # Assume we are in the context where classification with explanation is not required
        if not training_args.explainning:
            # Extract the first word after '[/INST]' marker, handling both upper and lower cases
            result_splitted = result[0]['generated_text'].split('[/INST]')[1]
            detected = re.search(pattern, result_splitted, flags=re.IGNORECASE)
            detected = detected.group(0) if detected else 'None'
            detected = detected.lower()
            predicted_labels.append(detected)
        else:
            # For classifying and explaining, extract the emotion after 'Emotion: '
            detected = 'None'
            result = result[0]['generated_text']
            result_splitted = result.split("Emotion: ")[2]
            match = re.search(pattern, result_splitted, flags=re.IGNORECASE)
            detected = match.group(0).lower() if match else 'None'  # Capitalize the detected emotion for consistency
            detected = detected.lower()
            predicted_labels.append(detected)

        
        print(detected)
        if label != detected:
            wrong_detection.append(str(result) + " THE TRUE LABEL IS "+ label)
            label_detection.append(detected)
        else:
            corrects += 1
    except:
        
        wrong_detection.append(str(result) + " THE TRUE LABEL IS " + label)
        label_detection.append(detected)
        exceptions += 1 
        
# Ensure that 'None' is included in both the actual and predicted labels
labels_extended = labels + ['none']  # This assumes 'labels' is a list of actual labels
labels_example = ["anger", "joy", "sadness", "guilt", "shame", "fear", "disgust", "none"]  # Include 'none' as a valid label

cm = confusion_matrix(labels, predicted_labels, labels=labels_example)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_example)
disp.plot(cmap=plt.cm.Blues)

# Define the directory
dir_name = "workspace/data/confusion_matrix/"

# Check if the directory exists
if not os.path.exists(dir_name):
    # If not, create the directory
    os.makedirs(dir_name)

# Now you can save the figure
plt.savefig(dir_name + 'confusion_matrix.png')  # Corrected file path to a writable directory

print("Confusion matrix saved as 'confusion_matrix.png'.")
print(f"Accuracy: {corrects / len(texts)}")

#save the artifact confusion matrix
mlflow.log_artifact(dir_name + 'confusion_matrix.png')



mlflow.log_metric("accuracy", corrects/len(texts))
mlflow.log_metric("exceptions", exceptions)
mlflow.log_metric("corrects", corrects)
mlflow.log_metric("wrong", len(texts) - corrects)
mlflow.log_metric("total", len(texts))
mlflow.end_run()


