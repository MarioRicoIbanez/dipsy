# -*- coding: utf-8 -*-
"""RoBERTa_MODEL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PKyqfyIDDjmSLGYXBst36Gt79kIvqCq-

# Modelo RoBERTa (Robust BERT)
"""
import os 

os.system('pip install transformers scikit-learn python-telegram-bot datasets deep_translator')

"""## Importando librerías necesarias y estableciendo parámetros globales"""

# Commented out IPython magic to ensure Python compatibility.
#Añadiendo las librerías necesarias para el encoding de los datos.
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer, RobertaModel, get_linear_schedule_with_warmup

from utils import *

import traceback
import asyncio
import os
import warnings
import re
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""GLOBAL PARAMS"""
MAX_LEN = 240
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS_KFOLD_FROZEN = 100
LEARNING_RATE_KFOLD_FROZEN = 3e-4
K_FOLDS_FROZEN = 2
BATCH_SIZE = 16
LOSS_FUNCTION = nn.CrossEntropyLoss()

"DIRECTORIES"
SAVE_DIRECTORY_FROZEN = "./MODELS/RoBERTa_entrenado_kfold"
SAVE_DIRECTORY_UNFROZEN = "./MODELS/RoBERTa_entrenado_kfold_unfrozen_last_layer"

TYPED_STORAGE_WARNING = re.compile(".TypedStorage is deprecated.")
FALLBACK_KERNEL_WARNING = re.compile(".Using FallbackKernel: aten.cumsum.")
TRITON_RANDOM_WARNING = re.compile(".using triton random, expect difference from eager.")

def warning_filter(message, category, filename, lineno, file=None, line=None):
    if category == UserWarning and (TYPED_STORAGE_WARNING.match(str(message)) or FALLBACK_KERNEL_WARNING.match(str(message)) or TRITON_RANDOM_WARNING.match(str(message))):
        return False
    return True


warnings.showwarning = warning_filter
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.ir")


# Establecer la semilla para PyTorch en las operaciones CPU
torch.manual_seed(42)

# Establecer la semilla para PyTorch en las operaciones en GPU
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(42)
# %matplotlib inline
# %load_ext tensorboard
# %tensorboard --logdir=./runs --host localhost --port 6006

"""## Inicio entrenamiento
Primero comprobamos que la gráfica se encuentra disponible para poder usarla
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

"""Descargamos y procesamos los datos"""

#Create DatasetDict
from datasets import DatasetDict, Dataset, load_dataset
dataset_path = 'RikoteMaster/isear_augmented'
dataset_dict = load_dataset(dataset_path)

dataset_dict = dataset_dict.remove_columns('Augmented')
dataset_dict

df_train = dataset_dict['train'].to_pandas()
df_test = dataset_dict['test'].to_pandas()
df_val = dataset_dict['validation'].to_pandas()

df_train

"""Cargamos el tokenizador para pasar las frases a tokens (unidad entendible por el modelo que vamos a estudiar)"""

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# Tokenize and encode the input data
def tokenize_and_encode(df):
  return tokenizer.batch_encode_plus(
    df.values.tolist(),
    add_special_tokens=True,
    padding='max_length',
    truncation=True,
    max_length=MAX_LEN,
    return_attention_mask=True,
    return_tensors='pt')

tokenized_features_train = tokenize_and_encode(df_train['Text_processed'])
tokenized_features_val = tokenize_and_encode(df_val['Text_processed'])
tokenized_features_test = tokenize_and_encode(df_test['Text_processed'])

print(tokenized_features_train.input_ids.shape)
print(tokenized_features_train.attention_mask.shape)

"""Observamos el diccionario en el que se encuentran los input_ids, es decir la codificación de cada palabra y la máscara de atención tan importante en los modelos basados en transformers.

Ahora hacemos one hot encoding de cada input_ids
"""

one_hot_encoder = OneHotEncoder()
target_one_hot_train = one_hot_encoder.fit_transform(df_train['Emotion'].values.reshape(-1, 1)).toarray()
target_one_hot_val = one_hot_encoder.fit_transform(df_val['Emotion'].values.reshape(-1, 1)).toarray()
target_one_hot_test = one_hot_encoder.fit_transform(df_test['Emotion'].values.reshape(-1, 1)).toarray()

target_one_hot_train.shape

train_inputs = tokenized_features_train["input_ids"]
val_inputs = tokenized_features_val["input_ids"]
test_inputs = tokenized_features_test["input_ids"]
train_labels = target_one_hot_train
val_labels = target_one_hot_val
test_labels = target_one_hot_test
train_masks = tokenized_features_train["attention_mask"]
val_masks = tokenized_features_val["attention_mask"]
test_masks = tokenized_features_test["attention_mask"]


train_inputs = torch.cat([train_inputs, val_inputs], dim=0)
train_labels = np.concatenate([train_labels, val_labels], axis=0)  # asumiendo que las etiquetas están en formato numpy
train_masks = torch.cat([train_masks, val_masks], dim=0)

train_inputs, train_labels, train_masks

print(train_inputs.shape, train_masks.shape, train_labels.shape)
print(test_inputs.shape, test_masks.shape, test_labels.shape)

"""Cargamos el modelo que vamos a usar"""

base_model = RobertaModel.from_pretrained("roberta-base")

"""Ahora congelamos las capas ya que primero vamos a hacer un entrenamiento en el que el modelo estará congelado y actuará como extractor de características. Y añadiremos una última capa de neuronas en las que si que estaremos clasificando aprovechando el token de salida."""

for param in base_model.parameters():
    param.requires_grad = False

    # Initialize the custom RoBERTa model
model = CustomRoBERTa(base_model.to(device), num_classes=len(set(df_train['Emotion'])))

os.system('nvidia-smi')

"""Aprovechamos las ventajas que nos da el torch 2.0, que tiene mejoras en entrenamientos de transformers usando la función compile"""

compiled_model = torch.compile(model)

"""Selección del optimizer"""

optimizer = torch.optim.AdamW(compiled_model.parameters(), LEARNING_RATE_KFOLD_FROZEN)

"""Ahora vamos a utilizar el warmup, Esta función se utiliza para crear un programador de velocidad de aprendizaje lineal con fase de calentamiento (warm-up) para el entrenamiento del modelo."""

total_steps = len(train_inputs) * EPOCHS_KFOLD_FROZEN
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                               num_training_steps=total_steps)

"""Procedemos a realizar el entrenamiento kfold, técnica utilizada con el fin de comprobar que nuestro modelo funciona bien ante diferentes separaciones de datos."""

epochs = EPOCHS_KFOLD_FROZEN
lr = LEARNING_RATE_KFOLD_FROZEN
kfold_results, model_frozen = kfold_cross_validation(train_inputs, train_labels, train_masks, model, device, epochs, lr, k_folds=2, batch_size=16,
                           patience=10)

for i, fold_result in enumerate(kfold_results[:-1]):
        message = f"Fold {i + 1} results (frozen model):\nAccuracy: {fold_result['validation_accuracies'][-1]:.4f}\nLoss: {fold_result['validation_losses'][-1]:.4f}"
print(message)

all_training_losses_frozen = []
all_training_accuracies_frozen = []
all_validation_losses_frozen = []
all_validation_accuracies_frozen = []

# Iterate through the results of each fold
for fold_result in kfold_results:
    all_training_losses_frozen.append(fold_result['training_losses'])
    all_training_accuracies_frozen.append(fold_result['training_accuracies'])
    all_validation_losses_frozen.append(fold_result['validation_losses'])
    all_validation_accuracies_frozen.append(fold_result['validation_accuracies'])

def plot_results(training_losses, validation_losses, loss_or_accuracy, title):
    import matplotlib.pyplot as plt

    for i, (t_loss, v_loss) in enumerate(zip(training_losses, validation_losses)):
        plt.plot(t_loss, label=f'Fold {i + 1} - Training ' + loss_or_accuracy)
        if v_loss is not None:
            plt.plot(v_loss, label=f'Fold {i + 1} - Validation ' + loss_or_accuracy)

    plt.xlabel('Epochs')
    plt.ylabel(loss_or_accuracy)
    plt.legend()
    plt.title(title)
    plt.savefig(title+'.png')

plot_results(all_training_losses_frozen, all_validation_losses_frozen, 'Loss', 'Trainning and Validation Losses for each fold')

plot_results(all_training_accuracies_frozen, all_validation_accuracies_frozen, 'Accuracy', 'Trainning and Validation Accuracies for each fold')



"""Guardamos el modelo para posteriormente hacer un entrenamiento en el que tengamos la última capa del mismo descongelado"""

model_frozen.base_model.save_pretrained(SAVE_DIRECTORY_FROZEN)

classifier_state_dict_path = os.path.join(SAVE_DIRECTORY_FROZEN, "classifier_state_dict.pt")
torch.save(model_frozen.classifier.state_dict(), classifier_state_dict_path)

"""Comprobamos los datos con el dataset de test para verificar divergencias"""

test_dataloader = create_dataloader(torch.tensor(test_inputs), torch.tensor(test_masks), torch.tensor(test_labels),
                                        BATCH_SIZE, device)

test_losses_frozen, test_accuracy_frozen, _, _ = test_model(model_frozen, test_dataloader, LOSS_FUNCTION)

"""## Ahora realizamos el entrenamiento descongelando etapas del modelo RoBERTa"""

EPOCHS_KFOLD_UNFROZEN = 5
LEARNING_RATE_KFOLD_UNFROZEN = 5e-5
K_FOLDS_UNFROZEN = 5
LAYERS_TO_UNFREEZE = -1

classifier_state_dict_path = os.path.join(SAVE_DIRECTORY_FROZEN, "classifier_state_dict.pt")
classifier_state_dict = torch.load(classifier_state_dict_path)

base_model = RobertaModel.from_pretrained("roberta-base")
for param in base_model.encoder.layer[:LAYERS_TO_UNFREEZE].parameters():
    param.requires_grad = True

unfrozen_model = CustomRoBERTa(base_model.to(device), num_classes=len(set(df_train['Emotion'])))
# Load the frozen classifier state dict into the unfrozen model
unfrozen_model.classifier.load_state_dict(classifier_state_dict)
# Set the optimizer and scheduler
optimizer = torch.optim.AdamW(unfrozen_model.parameters(), lr=LEARNING_RATE_KFOLD_UNFROZEN)
total_steps = len(train_inputs) * EPOCHS_KFOLD_UNFROZEN
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                            num_training_steps=total_steps)

compiled_model = torch.compile(unfrozen_model)
# Perform k-fold cross-validation on the unfrozen model
kfold_results_unfrozen, model_unfrozen_last_layer = kfold_cross_validation(train_inputs, train_labels, train_masks,
                                                                           compiled_model, device,
                                                                           EPOCHS_KFOLD_UNFROZEN,
                                                                           lr=LEARNING_RATE_KFOLD_UNFROZEN,
                                                                           k_folds=K_FOLDS_UNFROZEN,
                                                                           batch_size=BATCH_SIZE,
                                                                           patience=2)

# Clear unused objects and free up memory
del unfrozen_model
torch.cuda.empty_cache()

model_unfrozen_last_layer.base_model.save_pretrained(SAVE_DIRECTORY_UNFROZEN)
classifier_state_dict_path = os.path.join(SAVE_DIRECTORY_UNFROZEN, "classifier_state_dict.pt")
torch.save(model_unfrozen_last_layer.classifier.state_dict(), classifier_state_dict_path)

"""### Generación de la matriz de confusión, primero cargamos el modelo necesario vram muy ocupada"""

classifier_state_dict_path = os.path.join(SAVE_DIRECTORY_UNFROZEN, "classifier_state_dict.pt")
classifier_state_dict = torch.load(classifier_state_dict_path)

base_model = RobertaModel.from_pretrained("roberta-base")

model_unfrozen_last_layer = CustomRoBERTa(base_model.to(device), num_classes=len(set(df['Emotion'])))
# Load the frozen classifier state dict into the unfrozen model
model_unfrozen_last_layer.classifier.load_state_dict(classifier_state_dict)


model_unfrozen_last_layer = model_unfrozen_last_layer.to(device)
model_unfrozen_last_layer = torch.compile(model_unfrozen_last_layer)

from torch.utils.data import TensorDataset, DataLoader

# Define el tamaño del batch
batch_size = 16

# Crea un DataLoader para el conjunto de prueba
test_dataloader = create_dataloader(torch.tensor(test_inputs), torch.tensor(test_masks), torch.tensor(test_labels),
                                        BATCH_SIZE, device)
# Haz predicciones en mini-batches
preds = []
true_labels = []
model_unfrozen_last_layer.eval()
for batch in test_dataloader:
    b_input_ids, b_input_mask, b_labels = batch

    # Mueve los tensores al dispositivo correcto
    b_input_ids = b_input_ids
    b_input_mask = b_input_mask
    b_labels = b_labels

    with torch.no_grad():
        outputs = model_unfrozen_last_layer(b_input_ids, b_input_mask)
        logits = outputs.logits

        _, batch_preds = torch.max(logits, dim=1)


    preds.extend(batch_preds.cpu().numpy())
    true_labels.extend(b_labels.cpu().numpy())




# Obtén las categorías de tu objeto OneHotEncoder
categories = one_hot_encoder.categories_[0]

# Utiliza estas categorías para decodificar tus predicciones
preds_labels = categories[preds].reshape(-1,1)

true_labels = one_hot_encoder.inverse_transform(true_labels)

conf_matrix = confusion_matrix(true_labels, preds_labels
)

print(conf_matrix)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Crear el dataframe para seaborn.
confusion_df = pd.DataFrame(conf_matrix, index=categories, columns=categories)

plt.figure(figsize=(10,8))
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

def predict_emotion(sequence):
    # Tokeniza la secuencia
    inputs = tokenizer.encode_plus(
        sequence,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    # Obtiene los tensores de entrada y máscara de atención
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Evalúa la secuencia con el modelo
    with torch.no_grad():
        outputs = model_unfrozen_last_layer(input_ids, attention_mask)
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)

    # Decodifica las predicciones
    preds_labels = np.array(categories[preds]).reshape(-1,1)

    return preds_labels[0][0]

sequence = ""
emotion = predict_emotion(sequence)
print(f'The predicted emotion for "{sequence}" is {emotion}.')

"""### Futuras implementaciones
Aumento de dataset, entrenamiento con modelos menos científicos para conseguir que funcione de manera más aproximada
"""

