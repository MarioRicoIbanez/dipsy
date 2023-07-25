import os
import pandas as pd
import numpy as np
import torch
from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import PeftModel, PeftConfig, get_peft_config, get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, f1_score

# Semillas
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

#Create DatasetDict
dataset_path = 'RikoteMaster/isear_augmented'
dataset_dict = load_dataset(dataset_path)
dataset_dict = dataset_dict.remove_columns('Augmented')

#charge the train datasetDict as a df
df = dataset_dict['train'].to_pandas()

#create id2label and label2id
id2label = {i: label for i, label in enumerate(df['Emotion'].unique())}
label2id = {label: i for i, label in enumerate(df['Emotion'].unique())}

#apply label2id to the datasetDict
dataset_dict = dataset_dict.map(lambda example: {'labels': label2id[example['Emotion']]}, remove_columns=['Emotion'])

"""### Carga del tokenizador"""

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize_text(examples):
    return tokenizer(examples["Text_processed"], padding="max_length")

dataset_dict = dataset_dict.map(tokenize_text, batched=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'eval_accuracy': acc, 'f1': f1}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carga el mejor modelo guardado durante el entrenamiento
peft_model_id = "RikoteMaster/Bert_best_params"
config = PeftConfig.from_pretrained(peft_model_id)
inference_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=7,  id2label=id2label, label2id=label2id).to(device)
model = PeftModel.from_pretrained(inference_model, peft_model_id)

# Crea el Trainer con los argumentos y el modelo correctos
args = TrainingArguments(
    output_dir='./',  # no necesitamos guardar los modelos aquí, así que ponemos un directorio dummy
    do_train=False,   # no entrenamiento
    do_eval=True,     # si evaluación
    per_device_eval_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=args,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Realizar la evaluación
metrics = trainer.evaluate(dataset_dict['test'])
print(metrics)
