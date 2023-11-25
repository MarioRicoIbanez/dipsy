import re
import warnings
import os
import asyncio
import traceback
from utils import *
from transformers import AutoTokenizer, RobertaModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import mlflow


# Añadiendo las librerías necesarias para el encoding de los datos.



os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#set mlflow environment

mlflow.set_tracking_uri("http://localhost:8000")
mlflow.set_experiment("RoBERTa")
mflow.start_run("Entrenamiento base RoBERTa con dataset más grande")

"""GLOBAL PARAMS"""
MAX_LEN = 340
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 100
LEARNING_RATE = 3e-4
K_FOLDS = 1
BATCH_SIZE = 16
DATASET_NAME = "RikoteMaster/Emotion_Recognition_4_llama2_chat"
LOSS_FUNCTION = nn.CrossEntropyLoss()
TYPE = "frozen_base"


#log params mlflow as a dictionary 
mlflow.log_param({"MAX_LEN":MAX_LEN, "TEST_SIZE":TEST_SIZE, "RANDOM_STATE":RANDOM_STATE, 
                "EPOCHS":EPOCHS, "LEARNING_RATE":LEARNING_RATE, "K_FOLDS":K_FOLDS, 
                "BATCH_SIZE":BATCH_SIZE, "DATASET_NAME":DATASET_NAME, "TYPE":TYPE})



"DIRECTORIES"
SAVE_DIRECTORY_FROZEN = "./MODELS/RoBERTa_entrenado_base"

TYPED_STORAGE_WARNING = re.compile(".TypedStorage is deprecated.")
FALLBACK_KERNEL_WARNING = re.compile(".Using FallbackKernel: aten.cumsum.")
TRITON_RANDOM_WARNING = re.compile(
    ".using triton random, expect difference from eager.")


def warning_filter(message, category, filename, lineno, file=None, line=None):
    if category == UserWarning and (TYPED_STORAGE_WARNING.match(str(message)) or FALLBACK_KERNEL_WARNING.match(str(message)) or TRITON_RANDOM_WARNING.match(str(message))):
        return False
    return True


warnings.showwarning = warning_filter
warnings.filterwarnings("ignore", category=UserWarning,
                        module="torch._inductor.ir")


# Establecer la semilla para PyTorch en las operaciones CPU
torch.manual_seed(RANDOM_STATE)

# Establecer la semilla para PyTorch en las operaciones en GPU
torch.cuda.manual_seed(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(RANDOM_STATE)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from datasets import load_dataset


ds = load_dataset(DATASET_NAME, split="train")

print(ds)

ds['Text_processed']

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# Tokenize and encode the input data
tokenized_features = tokenizer.batch_encode_plus(ds['Text_processed'], add_special_tokens=True,
                                                 padding='max_length', truncation=True,
                                                 max_length=MAX_LEN, return_attention_mask=True,
                                                 return_tensors='pt'
                                                 )

one_hot_encoder = OneHotEncoder()
target_one_hot = one_hot_encoder.fit_transform(
    np.array(ds['Emotion']).reshape(-1, 1)).toarray()

print(target_one_hot.shape)

print(tokenized_features.keys())

train_inputs = tokenized_features['input_ids']
train_masks = tokenized_features['attention_mask']
train_labels = target_one_hot

print(train_inputs.shape, train_masks.shape, train_labels.shape)

base_model = RobertaModel.from_pretrained("roberta-base")

for param in base_model.parameters():
    param.requires_grad = False

    # Initialize the custom RoBERTa model
model = CustomRoBERTa(base_model.to(device), num_classes=len(set(ds['Emotion'])))

compiled_model = torch.compile(model)

optimizer = torch.optim.AdamW(compiled_model.parameters(), LEARNING_RATE)

total_steps = len(train_inputs) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                            num_training_steps=total_steps)

kfold_results, model_frozen = kfold_cross_validation(train_inputs, train_labels, train_masks, compiled_model, device, EPOCHS, lr=LEARNING_RATE,
                                                     k_folds=K_FOLDS,
                                                     batch_size=BATCH_SIZE)

for i, fold_result in enumerate(kfold_results):
    message = f"Fold {i + 1} results (frozen model):\nAccuracy: {fold_result['validation_accuracies'][-1]:.4f}\nLoss: {fold_result['validation_losses'][-1]:.4f}"
    print(message)

training_losses_frozen = []
training_accuracies_frozen = []
validation_losses_frozen = []
validation_accuracies_frozen = []



# Iterate through the results of each fold
for fold_result in kfold_results:
    training_losses_frozen.append(fold_result['training_losses'])
    training_accuracies_frozen.append(fold_result['training_accuracies'])
    validation_losses_frozen.append(fold_result['validation_losses'])
    validation_accuracies_frozen.append(
        fold_result['validation_accuracies'])

#LOG METRICS
mlflow.log_metrics({"training_losses_frozen":training_losses_frozen, "training_accuracies_frozen":training_accuracies_frozen, 
                    "validation_losses_frozen":validation_losses_frozen, "validation_accuracies_frozen":validation_accuracies_frozen})
                    
print(training_losses_frozen, training_accuracies_frozen)

import os
import matplotlib.pyplot as plt

def plot_and_save_results(training_losses, validation_losses, metric, title, file_name):
    # Create the directory ./Results if it does not exist
    if not os.path.exists('./Results'):
        os.makedirs('./Results')
    
    # Create a new figure
    plt.figure()
    
    # Check if losses are a list of lists (multiple folds)
    if all(isinstance(loss, list) for loss in training_losses):
        # Plot the loss or accuracy curve for training for each fold
        for i, losses in enumerate(training_losses):
            plt.plot(losses, label=f'Training {metric} Fold {i+1}')
        # Plot the loss or accuracy curve for validation for each fold
        for i, losses in enumerate(validation_losses):
            plt.plot(losses, label=f'Validation {metric} Fold {i+1}')
    else:
        # Plot the loss or accuracy curve for training
        plt.plot(training_losses, label=f'Training {metric}')
        # Plot the loss or accuracy curve for validation if available
        if validation_losses is not None:
            plt.plot(validation_losses, label=f'Validation {metric}')

    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.title(title)
    
    # Save the figure in the directory ./Results with the provided name
    plt.savefig(f'./Results/{file_name}.png')
    
    # Close the figure to free up memory
    plt.close()

# Example usage of the function
plot_and_save_results(
    training_losses_frozen, 
    validation_losses_frozen,
    'Loss', 
    'Training and Validation Losses', 
    'training_validation_loss'
)




model_frozen.base_model.save_pretrained(SAVE_DIRECTORY_FROZEN)

classifier_state_dict_path = os.path.join(
    SAVE_DIRECTORY_FROZEN, "classifier_state_dict.pt")
classifier_state_dict = torch.load(classifier_state_dict_path)
