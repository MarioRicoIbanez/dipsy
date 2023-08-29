#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
#
#  ### Creating synthetic dataset
#  The process is going to be, create a good prompt for llama-2-40b-chat-hf. And infere over the 63k prompt in order to make a emotion-reason dataset. By this way, we can try to make a LLM that is capable to explain emotions and may make better predictions
#  

# %%

from numpy import save, asarray
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


# %%





# The model that you want to train from the Hugging Face hub
model_name = "meta-llama/Llama-2-13b-chat-hf"

# The instruction dataset to use
dataset_name = "RikoteMaster/Emotion_Recognition_4_llama2_chat"

# Fine-tuned model name
new_model = "llama-2-7b-sentiment-analyzer"

device_map = {"": 0}

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
# Loading model
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


# %%


from datasets import load_dataset
import pandas as pd

ds = load_dataset("RikoteMaster/Emotion_Recognition_4_llama2_chat")
ds = ds['train']
ds = pd.DataFrame.from_dict(ds)

print(ds)

def bigger_formatting(text, label):
    prompt = f"""<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows:

    Anger, Joy, Sadness, Guilt, Shame, fear or disgust. 
    
    Sentence: {text} Emotion: {label} [/INST] Please answer only with the explanation of the Emotion. For example, in the input Sentence: I feel sad when my mother leaves home. Emotion: Sadness. You should answer EXPLANATION: In this sentence the feeling of sadness is due to the person is not going to see her mother in a period of time. """
    return prompt

for index, row in ds.iterrows():
    ds.loc[index, 'text'] = bigger_formatting(row['Text_processed'], row['Emotion'])

print(ds['text'][0])


# %%


from tqdm import tqdm

def prediction(text):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=700)
    result = pipe(text)
    return result[0]['generated_text']

predictions = []

# Set the total number of iterations (progress total)
total_iterations = len(ds)

# Create a tqdm progress bar
for index, row in tqdm(ds.iterrows(), total=total_iterations, desc="Generating Predictions"):
    prediction_aux = prediction(row['text'])
    predictions.append(prediction_aux)
    if index%100 == 0:
        print(prediction_aux)
        save('data_aux.npy', asarray(predictions))

from numpy import asarray
from numpy import save, load

#save data
save('data.npy', asarray(predictions))

predicted = load('data.npy')
predicted[0]


# %%


# save numpy array as npy file
from numpy import asarray
from numpy import save, load

#save data
save('data.npy', asarray(predictions))

predicted = load('data.npy')
predicted[0]


# %% [markdown]
# ### Analyzing quality of explanations

# %%
from numpy import save, load
from datasets import load_dataset
import pandas as pd

predicted = load('data.npy')
ds = load_dataset("RikoteMaster/Emotion_Recognition_4_llama2_chat")
ds = ds['train']
ds = pd.DataFrame.from_dict(ds)
print(len(ds),len(predicted))



# %%
ds_Text_processed = ds['Text_processed']
ds_Emotion = ds['Emotion']

# %%
counter = 0
counter_0 = 0
explanation_concat = []
i = 0 
axis_drop = []
for sentence in predicted:
    try:
        explanation = sentence.lower().split('explanation:')[2]
        if len(explanation) == 0 :
            counter_0 += 1
            axis_drop.append(i)

        else: 
            explanation_concat.append(explanation)
            
    except:
        counter += 1
        axis_drop.append(i)

    i += 1

print(counter, counter_0, i) 

# %%
ds_Text_processed = ds_Text_processed.drop(axis_drop).reset_index(drop=True)
ds_Emotion = ds_Emotion.drop(axis_drop).reset_index(drop=True)

print(len(ds_Text_processed), len(explanation_concat))

# %%
explanation_cleaned_length = []
counter_10 = 0
axis_drop = []
for i in range(0,len(explanation_concat)): 
    if len(explanation_concat[i].split()) >= 10:
        explanation_cleaned_length.append(explanation_concat[i])
    else:
        counter_10 += 1
        axis_drop.append(i)
        
ds_Text_processed = ds_Text_processed.drop(axis_drop).reset_index(drop=True)
ds_Emotion = ds_Emotion.drop(axis_drop).reset_index(drop=True)

print(len(ds_Text_processed), len(explanation_cleaned_length))

# %%
counter_is = 0
explanation_cleaned_is = []

i = 0
axis_drop = []

for sentence in explanation_cleaned_length:
    if 'is:' in sentence:
        counter_is += 1
        axis_drop.append(i)
    else: 
        explanation_cleaned_is.append(sentence)
    i += 1

ds_Text_processed = ds_Text_processed.drop(axis_drop).reset_index(drop=True)
ds_Emotion = ds_Emotion.drop(axis_drop).reset_index(drop=True)

print(len(ds_Text_processed), len(explanation_cleaned_is))

# %%
for i in range(0, len(explanation_cleaned_is),100):
    print(explanation_cleaned_is[i])

# %%
counter_emotion = 0
explanation_cleaned_emotion = []

i = 0
axis_drop = []

for sentence in explanation_cleaned_is:
    if 'emotion:' in sentence:
        counter_emotion += 1
        axis_drop.append(i)
    else: 
        explanation_cleaned_emotion.append(sentence)
    i += 1
        
        
ds_Text_processed = ds_Text_processed.drop(axis_drop).reset_index(drop=True)
ds_Emotion = ds_Emotion.drop(axis_drop).reset_index(drop=True)

print(len(ds_Text_processed), len(explanation_cleaned_emotion))



# %% [markdown]
# ### Now creating a new dataset with the new formatting

# %%
ds_explanation = pd.DataFrame(explanation_cleaned_emotion, columns=['Explanation'])

# Create a mask for rows that start with '\n'
mask = ~ds_explanation['Explanation'].str.startswith('\n')

# Apply the mask to the DataFrames
ds_explanation = ds_explanation[mask]
ds_Text_processed = ds_Text_processed[mask]  # Apply the same mask to ds_Text_processed
ds_Emotion = ds_Emotion[mask]  # Apply the same mask to ds_Emotion

# %%
ds_explanation = ds_explanation.reset_index(drop=True)
ds_Text_processed = ds_Text_processed.reset_index(drop=True)
ds_Emotion = ds_Emotion.reset_index(drop=True)

# %%
ds_explanation['Text_processed'] = ds_Text_processed
ds_explanation['Emotion'] = ds_Emotion

# %%
ds_explanation


# %%
def bigger_formatting(text, emotion, explanation):
    formatted_text = f"""<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows:

    Anger, Joy, Sadness, Guilt, Shame, Fear, or Disgust
    
    You have to first classify the emotion using Emotion: and later, you will have to provide an explanation of why you think the sentence is expressing that emotion, using Explanation:

    Sentence: {text} [/INST] Emotion: {emotion} Explanation: {explanation} <s>"""
    
    return formatted_text

# Apply the formatting function to each row and add to a list
formatted_rows = [bigger_formatting(text, emotion, explanation) for text, emotion, explanation in zip(ds_Text_processed, ds_Emotion, ds_explanation['Explanation'])]

# Add the list as a new column 'Formatted_Text' to the ds_explanation DataFrame
ds_explanation['text'] = formatted_rows

# %%
ds_explanation['text'][0]

# %%
from datasets import Dataset
ds_explanation = Dataset.from_pandas(ds_explanation)
ds_explanation.push_to_hub('RikoteMaster/llama2_classifying_and_explainning_v2')

# %% [markdown]
# ### Re-trainning with explanation

# %%
# !pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7


# %%
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

# %%
# The model that you want to train from the Hugging Face hub
model_name = "meta-llama/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "RikoteMaster/llama2_classifying_and_explainning_v2"

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
output_dir = "./results"

# Number of training epochs
num_train_epochs = 4

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

# %%
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
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
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

# %%
dataset['text'][0]

# %%
sentence = "I hate football"
text = f"""<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows:\n\n    Anger, Joy, Sadness, Guilt, Shame, Fear, or Disgust\n    \n    You have to first classify the emotion using Emotion: and later, you will have to provide an explanation of why you think the sentence is expressing that emotion, using Explanation:\n\n    Sentence: {sentence} [/INST]"""
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(text)
print(result)

# %% [markdown]
# Making inference in order to try if the reasonning concept improves the accuracy

# %% [markdown]
# ### Inference over isear validation

# %%
from datasets import load_dataset

ds = load_dataset("RikoteMaster/isear_for_llama2")

texts = ds['validation']['Text_processed']
labels = ds['validation']['Emotion']


# %%
wrong_detection = []
corrects = 0
exceptions = 0
for sentence, label in zip(texts, labels):
    text = f"""<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows:\n\n    Anger, Joy, Sadness, Guilt, Shame, Fear, or Disgust\n    \n    You have to first classify the emotion using Emotion: and later, you will have to provide an explanation of why you think the sentence is expressing that emotion, using Explanation:\n\n    Sentence: {sentence} [/INST]"""
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=len(tokenizer(text)) + 200)
    result = pipe(text)
    try:
        detected = result[0]['generated_text'].split('Emotion:')[2].split()[0]
        if label != detected:
            wrong_detection.append(str(result) + " THE TRUE LABEL IS "+ label)
        else:
            corrects += 1
    except:
        
        wrong_detection.append(str(result) + " THE TRUE LABEL IS " + label)
        exceptions += 1 
        


print(corrects/len(texts))

# %%
print(exceptions)

# %%
for item in wrong_detection:
    print(item)

# %% [markdown]
# ### Inference over isear test

# %%
from datasets import load_dataset

ds = load_dataset("RikoteMaster/isear_for_llama2")

texts = ds['test']['Text_processed']
labels = ds['test']['Emotion']



wrong_detection = []
corrects = 0
exceptions = 0
for sentence, label in zip(texts, labels):
    text = f"""<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows:\n\n    Anger, Joy, Sadness, Guilt, Shame, Fear, or Disgust\n    \n    You have to first classify the emotion using Emotion: and later, you will have to provide an explanation of why you think the sentence is expressing that emotion, using Explanation:\n\n    Sentence: {sentence} [/INST]"""
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=len(tokenizer(text)) + 200)
    result = pipe(text)
    try:
        detected = result[0]['generated_text'].split('Emotion:')[2].split()[0]
        if label != detected:
            wrong_detection.append(str(result) + " THE TRUE LABEL IS "+ label)
        else:
            corrects += 1
    except:
        
        wrong_detection.append(str(result) + " THE TRUE LABEL IS " + label)
        exceptions += 1 
        


print(corrects/len(texts))


# %% [markdown]
# ### CONCLUSIONS

# %% [markdown]
# After seeing the results, I think that the not stabilized dataset could create problems in detecting emotions, so the next step is stabilize the classes of the dataset. 
#
# What is supposed to happen to the model when it has an unstable dataset?
#
# I think that the result on the inference may be more probably for the classes with more samples in the dataset.

# %% [markdown]
# ### Retrainning with dataset cleaned

# %% [markdown]
# In the process of cleaning the dataset, I have selected the largest 2884 sentences from each class so as to balance the classes and choose the higher-quality sentences.

# %%

# %% [markdown]
# # DO NOT RUN ONLY ONCE

# %%
from datasets import load_dataset

ds = load_dataset("RikoteMaster/llama2_classifying_and_explainning_v2")
ds = ds['train']
ds = pd.DataFrame.from_dict(ds)

import pandas as pd

ds_pandas = pd.DataFrame.from_dict(ds['train'])

ds_pandas.Emotion.value_counts()

import pandas as pd

# Assuming ds_pandas is your dataset

# Create a new DataFrame to store the reduced dataset
reduced_dataset = pd.DataFrame(columns=ds_pandas.columns)

# Iterate over each emotion
for emotion in ds_pandas['Emotion'].unique():
    # Get the subset of the dataset for the current emotion
    subset = ds_pandas[ds_pandas['Emotion'] == emotion]

    # Sort the subset by the 'Text_processed' column in descending order
    sorted_subset = subset.sort_values(by='Text_processed', ascending=False)

    # Select the top 2884 samples from the sorted subset
    selected_samples = sorted_subset.head(2884)

    # Append the selected samples to the reduced_dataset
    reduced_dataset = reduced_dataset.append(selected_samples, ignore_index=True)

# The reduced_dataset now contains 2884 samples for each emotion, with the largest Text_processed values

import pandas as pd

# Assuming ds_pandas is your dataset

# Remove duplicated sentences
ds_pandas_unique = ds_pandas.drop_duplicates(subset='Text_processed')

# Get value counts with Emotion label
value_counts_by_emotion = ds_pandas_unique['Emotion'].value_counts()

print(value_counts_by_emotion)

import pandas as pd

# Assuming ds_pandas is your dataset

# Create a new DataFrame to store the selected sentences
selected_sentences = pd.DataFrame(columns=ds_pandas.columns)

# Iterate over each emotion
for emotion in ds_pandas_unique['Emotion'].unique():
    # Get the subset of the dataset for the current emotion
    subset = ds_pandas_unique[ds_pandas_unique['Emotion'] == emotion]

    # Sort the subset by the 'Text_processed' column in descending order
    sorted_subset = subset.sort_values(by='Text_processed', ascending=False)

    # Select the top 2000 sentences from the sorted subset
    selected_samples = sorted_subset.head(2000)

    # Append the selected samples to the selected_sentences DataFrame
    selected_sentences = selected_sentences.append(selected_samples, ignore_index=True)

# The selected_sentences DataFrame now contains the 2000 largest sentences for each emotion

def bigger_formatting(text, label):
    prompt = f"""<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows:

    Anger, Joy, Sadness, Guilt, Shame, fear or disgust. 
    
    Sentence: {text} Emotion: {label} [/INST] Please answer only with the explanation of the Emotion. For example, in the input Sentence: I feel sad when my mother leaves home. Emotion: Sadness. You should answer EXPLANATION: In this sentence the feeling of sadness is due to the person is not going to see her mother in a period of time. """
    return prompt

for index, row in selected_sentences.iterrows():
    selected_sentences.loc[index, 'text'] = bigger_formatting(row['Text_processed'], row['Emotion'])

ds = Dataset.from_pandas(selected_sentences)

ds.push_to_hub('RikoteMaster/llama2_classifying_and_explainning_v4')


# %%

# %%
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

# %%
# The model that you want to train from the Hugging Face hub
model_name = "meta-llama/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "RikoteMaster/llama2_classifying_and_explainning_v4"

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
num_train_epochs = 7

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

# %%
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
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
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

# %%
from datasets import load_dataset

ds = load_dataset("RikoteMaster/isear_for_llama2")

texts = ds['test']['Text_processed']
labels = ds['test']['Emotion']


label_detection = []
wrong_detection = []
corrects = 0
exceptions = 0
for sentence, label in zip(texts, labels):
    text = f"""<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows:\n\n    Anger, Joy, Sadness, Guilt, Shame, Fear, or Disgust\n    \n    You have to first classify the emotion using Emotion: and later, you will have to provide an explanation of why you think the sentence is expressing that emotion, using Explanation:\n\n    Sentence: {sentence} [/INST]"""
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=len(tokenizer(text)) + 200)
    result = pipe(text)
    try:
        detected = result[0]['generated_text'].split('Emotion:')[2].split()[0]
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


# %%
print(set(label_detection))

# %%
sentence = "I hate beeing bad treated when I went shopping with my lovely sister"
text = f"""<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows:\n\n    Anger, Joy, Sadness, Guilt, Shame, Fear, or Disgust\n    \n    You have to first classify the emotion using Emotion: and later, you will have to provide an explanation of why you think the sentence is expressing that emotion, using Explanation:\n\n    Sentence: {sentence} [/INST]"""
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
result = pipe(text)
print(result)

# %%
