

#modifica estos installs y hazlos con la biblioteca  os
import os
import pandas as pd
import numpy as np
import torch

os.system('pip install datasets')
os.system('pip install transformers[torch]')
os.system('pip install accelerate>=0.20.1')
os.system('pip install peft')
os.system('pip install hugingface_hub')
os.system('huggingface-cli login --token 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')


# Semillas
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
from datasets import DatasetDict, Dataset, load_dataset

from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback, IntervalStrategy


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import torch
from sklearn.metrics import accuracy_score, f1_score
    



#Create DatasetDict
dataset_path = 'RikoteMaster/isear_augmented'
dataset_dict = load_dataset(dataset_path)

dataset_dict = dataset_dict.remove_columns('Augmented')
dataset_dict

#charge the train datasetDict as a df
df = dataset_dict['train'].to_pandas()
df.head()
#create id2label and label2id
id2label = {i: label for i, label in enumerate(df['Emotion'].unique())}
label2id = {label: i for i, label in enumerate(df['Emotion'].unique())}

#apply label2id to the datasetDict
dataset_dict = dataset_dict.map(lambda example: {'labels': label2id[example['Emotion']]}, remove_columns=['Emotion'])

dataset_dict

"""### Carga del tokenizador"""

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize_text(examples):
    return tokenizer(examples["Text_processed"], padding="max_length")

dataset_dict = dataset_dict.map(tokenize_text, batched=True)
dataset_dict



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')

    return {'eval_accuracy': acc, 'f1': f1}


device = 'cuda' if torch.cuda.is_available() else 'cpu'

peft_config  = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.8,
        bias="none",
        task_type=TaskType.SEQ_CLS
        )




def compute_objective(metrics):

    return metrics['eval_accuracy'] + metrics['f1']


batch_size = 16
epochs = 70

output_dir = './bert_best_params'
logging_steps = len(dataset_dict['train']) // batch_size
args = TrainingArguments( output_dir=output_dir, 
                        num_train_epochs=epochs,
                        learning_rate=0.000024509631236742206,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        weight_decay=8.393030941902047e-05,
                        evaluation_strategy = IntervalStrategy.STEPS,
                        eval_steps = 50, # Evaluation and Save happens every 50 steps
                        save_total_limit = 5,
                        logging_steps=logging_steps,
                        fp16=True,
                        push_to_hub=False,
                        load_best_model_at_end=True,
                        metric_for_best_model='accuracy')

model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=len(id2label), ignore_mismatched_sizes=True).to(device)
model = get_peft_model(model, peft_config)
model = model.to(device)
model = model.compile()

trainer = Trainer(model=model,
                  args=args,
                  train_dataset=dataset_dict['train'],
                  eval_dataset=dataset_dict['validation'],
                  compute_metrics=compute_metrics,
                  tokenizer=tokenizer,
                 callbacks = [EarlyStoppingCallback(early_stopping_patience=15)])


trainer.train()




model.push_to_hub("RikoteMaster/Bert_best_params")
trainer.push_to_hub("RikoteMaster/Bert_best_params_trainer")


