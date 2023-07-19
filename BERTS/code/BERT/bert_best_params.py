
#modifica estos installs y hazlos con la biblioteca  os
import os
import pandas as pd

os.system('pip install datasets')
os.system('pip install transformers[torch]')
os.system('pip install accelerate>=0.20.1')
os.system('pip install peft')
os.system('pip install hugingface_hub')




#Create DatasetDict
from datasets import DatasetDict, Dataset, load_dataset
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

from transformers import AutoTokenizer
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize_text(examples):
    return tokenizer(examples["Text_processed"], padding="max_length")

dataset_dict = dataset_dict.map(tokenize_text, batched=True)
dataset_dict

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import torch
from sklearn.metrics import accuracy_score, f1_score

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
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
        )

def model_init(trial):
    if 'model' in locals():
        del model
        torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=len(id2label), ignore_mismatched_sizes=True).to(device)
    model = get_peft_model(model, peft_config)


    return model


def compute_objective(metrics):

    return metrics['eval_accuracy'] + metrics['f1']


batch_size = 16
epochs = 70

output_dir = '../NASFolder/results_searching_hyperparameters'
logging_steps = len(dataset_dict['train']) // batch_size
args = TrainingArguments( output_dir=output_dir, 
                        num_train_epochs=epochs,
                        learning_rate=0.00024509631236742206,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        weight_decay=8.393030941902047e-05,
                        evaluation_strategy='epoch',
                        save_strategy='epoch',
                        logging_steps=logging_steps,
                        fp16=True,
                        push_to_hub=False,
                        load_best_model_at_end=True,
                        metric_for_best_model='accuracy')
model = model_init

trainer = Trainer(model=model,
                  args=args,
                  train_dataset=dataset_dict['train'],
                  eval_dataset=dataset_dict['validation'],
                  compute_metrics=compute_metrics,
                  tokenizer=tokenizer)


trainer.train()



from huggingface_hub import HfApi

# Assumes your Hugging Face API token is stored in an environment variable
token = "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"

trainer.push_to_hub("RikoteMaster/bert_best_params_lora_train", use_auth_token=token)