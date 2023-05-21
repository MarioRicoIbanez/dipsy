import pandas as pd
import numpy as np
import re
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from deep_translator import GoogleTranslator
import string
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from collections import namedtuple
import torch.nn as nn
from transformers import RobertaModel, AdamW, get_linear_schedule_with_warmup
import os
from sklearn.model_selection import KFold
import copy
import telegram
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.multiclass import unique_labels


# This function cleans the given text by removing punctuation marks, special characters, and converting to lowercase.

def clean_text(text):
    # to lower case
    text = text.lower()
    # remove links
    text = re.sub('https:\/\/\S+', '', text)
    # remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # remove next line
    text = re.sub(r'[^ \w\.]', '', text)
    # remove words containing numbers
    text = re.sub('\w*\d\w*', '', text)

    return text


def translate_text(text):
    traductor = GoogleTranslator(source='es', target='en')
    resultado = traductor.translate(text)
    return resultado


# This function loads and preprocesses data from a csv file, applies the clean_text function and adds a new column 'Text_processed'

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, names=['Emotion', 'Text', 'DNTKNOW']).drop(columns=['DNTKNOW']).dropna()
    df['Text_processed'] = df.Text.apply(clean_text)
    return df


# This function splits the tokenized features into training and validation sets

def split_data(features, labels, test_size=0.25, validation_size=0.2, random_seed=None):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba.
    """
    if random_seed is not None:
        random_state = random_seed
    else:
        random_state = np.random.RandomState(seed=None)

    # Divide los datos en conjuntos de entrenamiento y prueba
    train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks = train_test_split(
        features["input_ids"],
        labels,
        features["attention_mask"],
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # Divide los datos de entrenamiento en conjuntos de entrenamiento y validación
    train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = train_test_split(
        train_inputs,
        train_labels,
        train_masks,
        test_size=validation_size,
        random_state=random_state,
        stratify=train_labels,
    )

    return train_inputs, validation_inputs, test_inputs, torch.tensor(train_labels), torch.tensor(
        validation_labels), torch.tensor(test_labels), train_masks, validation_masks, test_masks

    # This function creates dataloaders using the given inputs, masks, labels, and batch size


def create_dataloader(inputs, masks, labels, batch_size, device):
    data = TensorDataset(inputs.to(device), masks.to(device), labels.to(device))
    dataloader = DataLoader(data, batch_size=batch_size)
    return dataloader


class CustomRoBERTa(nn.Module):
    def __init__(self, base_model, num_classes=2):  # Change the default value of num_classes to 7
        super(CustomRoBERTa, self).__init__()  # Inheritance of nn.Module overloading the constructor with CustomRoBERTa
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Create a named tuple to hold the logits
        ModelOutput = namedtuple("ModelOutput", ["logits"])
        return ModelOutput(logits=logits)

    @classmethod
    def from_pretrained(cls, model_path, num_classes):
        base_model = RobertaModel.from_pretrained(model_path)
        custom_model = cls(base_model, num_classes)
        classifier_state_dict = torch.load(os.path.join(model_path, "classifier_state_dict.pt"))
        custom_model.classifier.load_state_dict(classifier_state_dict)
        return custom_model


# This function trains the given model using the given training and validation dataloaders, optimizer, scheduler, loss function, device, and epochs
def train_model(model, train_dataloader, validation_dataloader, optimizer, scheduler, loss_function, device, epochs,
                patience=10):
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    epoch_early_stopping = 0

    best_validation_loss = float("inf")
    consecutive_no_improvement = 0

    for epoch_i in range(epochs):
        print(f'Training epoch {epoch_i + 1}')
        model.train()
        total_loss = 0
        total_train_accuracy = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            b_input_ids = batch[0]  # move input tensor to device
            b_input_mask = batch[1]  # move input tensor to device
            b_labels_one_hot = batch[2]  # move input tensor to device
            model.zero_grad()

            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits.detach().cpu().numpy()
            label_ids_one_hot = b_labels_one_hot.cpu().numpy()
            label_ids = np.argmax(label_ids_one_hot, axis=  0)

            loss = loss_function(outputs.logits, b_labels_one_hot)

            total_train_accuracy += flat_accuracy(logits, label_ids)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        training_losses.append(avg_train_loss)
        training_accuracies.append(avg_train_accuracy)

        print(
            f'Epoch {epoch_i + 1} - Average training loss: {avg_train_loss:.2f}, Average training accuracy: {avg_train_accuracy:.2f}')

        if validation_dataloader is None:  # If we are not providing the validation dataset, we are not going to validate the model, used in the last kfold division
            continue

        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0


        for batch in validation_dataloader:
            b_input_ids = batch[0]  # move input tensor to device
            b_input_mask = batch[1]  # move input tensor to device
            b_labels_one_hot = batch[2]  # move input tensor to device

            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask)

            logits = outputs.logits.detach().cpu().numpy()
            label_ids_one_hot = b_labels_one_hot.cpu().numpy()
            label_ids = np.argmax(label_ids_one_hot, axis=0)

            total_eval_accuracy += flat_accuracy(logits, label_ids)

            eval_loss = loss_function(outputs.logits, b_labels_one_hot)
            total_eval_loss += eval_loss.item()

        avg_eval_loss = total_eval_loss / len(validation_dataloader)
        avg_eval_accuracy = total_eval_accuracy / len(validation_dataloader)
        validation_losses.append(avg_eval_loss)
        validation_accuracies.append(avg_eval_accuracy)
        print(
            f'Epoch {epoch_i + 1} - Average validation loss: {avg_eval_loss:.2f}, Average validation accuracy: {avg_eval_accuracy:.2f}')

        # Early stopping logic
        if avg_eval_loss < best_validation_loss:
            best_validation_loss = avg_eval_loss
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
            if consecutive_no_improvement >= patience:
                print("Early stopping triggered, no improvement for {} consecutive epochs".format(patience))

                epoch_early_stopping = epoch_i

                break

    return training_losses, training_accuracies, validation_losses, validation_accuracies, epoch_early_stopping

    
   
# This function calculates the accuracy of the predicted labels compared to the actual labels


def plot_confusion_matrix_one_hot(y_true, y_pred, classes, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  # annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(classes);
    ax.yaxis.set_ticklabels(classes)
    plt.savefig('Confussion Matrix.png')


def test_model(model, test_dataloader, loss_function):
    model.eval()
    total_test_accuracy = 0
    total_test_loss = 0
    total_examples = 0

    y_true = []
    y_pred = []

    for batch in test_dataloader:
        b_input_ids, b_input_mask, b_labels_one_hot = batch

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs.logits.detach().cpu().numpy()
        label_ids_one_hot = b_labels_one_hot.to('cpu').numpy()
        label_ids = np.argmax(label_ids_one_hot, axis=1)

        y_true.extend(label_ids)
        y_pred.extend(np.argmax(logits, axis=1))

        total_test_accuracy += flat_accuracy(logits, label_ids) * len(label_ids)
        total_examples += len(label_ids)

        test_loss = loss_function(outputs.logits, b_labels_one_hot)
        total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_test_accuracy = total_test_accuracy / total_examples
    print(f'Average test loss: {avg_test_loss:.2f}, Average test accuracy: {avg_test_accuracy:.2f}')
    return avg_test_loss, avg_test_accuracy, y_true, y_pred


def kfold_cross_validation(train_inputs, train_labels, train_masks, model, device, epochs, lr, k_folds=5, batch_size=16,
                           patience=10):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, validation_idx) in enumerate(kfold.split(train_inputs, train_labels)):
        print(f"Fold {fold + 1}")

        # Obtén los datos de entrenamiento y validación para el fold actual
        train_inputs_fold = torch.as_tensor(train_inputs[train_idx])
        validation_inputs = torch.as_tensor(train_inputs[validation_idx])
        train_labels_fold = torch.as_tensor(train_labels[train_idx])
        validation_labels = torch.as_tensor(train_labels[validation_idx])
        train_masks_fold = torch.as_tensor(train_masks[train_idx])
        validation_masks = torch.as_tensor(train_masks[validation_idx])

        # Crea dataloaders para el fold actual
        train_dataloader = create_dataloader(train_inputs_fold, train_masks_fold, train_labels_fold, batch_size, device)
        validation_dataloader = create_dataloader(validation_inputs, validation_masks, validation_labels, batch_size,
                                                  device)

        # Copia el modelo para usar una versión nueva en cada fold
        model_copy = copy.deepcopy(model)
        model_copy.to(device)

        # Crea el optimizador, scheduler y función de pérdida para el fold actual
        optimizer = torch.optim.AdamW(model_copy.parameters(), lr=lr, eps=1e-8)

        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                                    num_training_steps=total_steps)
        loss_function = nn.CrossEntropyLoss()

        # Entrena y evalúa el modelo en el fold actual
        training_losses, training_accuracies, validation_losses, validation_accuracies, epoch_early_stopping = train_model(
            model_copy, train_dataloader, validation_dataloader, optimizer, scheduler, loss_function, device, epochs,
            patience=patience)

        # Guarda los resultados del fold
        fold_results.append({
            'training_losses': training_losses,
            'training_accuracies': training_accuracies,
            'validation_losses': validation_losses,
            'validation_accuracies': validation_accuracies
        })

    print("Final Training with all data")

    train_inputs_all = torch.tensor(train_inputs)
    train_labels_all = torch.tensor(train_labels)
    train_masks_all = torch.tensor(train_masks)

    train_dataloader = create_dataloader(train_inputs_all, train_masks_all, train_labels_all, batch_size, device)

    final_model = copy.deepcopy(model)
    final_model.to(device)

    optimizer = AdamW(final_model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                                num_training_steps=total_steps)
    loss_function = nn.CrossEntropyLoss()

    # Entrena el modelo final con todos los datos de entrenamiento y sin validation_dataloader
    training_losses, training_accuracies, _, _, _ = train_model(
        final_model, train_dataloader, None, optimizer, scheduler, loss_function, device,
        epochs if epoch_early_stopping == 0 else epoch_early_stopping)

    fold_results.append({
        'training_losses': training_losses,
        'training_accuracies': training_accuracies,
        'validation_losses': None,
        'validation_accuracies': None
    })

    return fold_results, final_model


def flat_accuracy(preds, labels_one_hot):
    pred_flat = np.argmax(preds, axis=0).flatten()
    labels_flat = np.argmax(labels_one_hot, axis=0).flatten()  # Modify this line to handle one-hot encoded labels
    output = np.sum(pred_flat == labels_flat) / len(labels_flat)

    return output


# Define the custom RoBERTa model class

def save_and_plot_results(training_losses, validation_losses, prefix):
    import matplotlib.pyplot as plt

    for i, (t_loss, v_loss) in enumerate(zip(training_losses, validation_losses)):
        plt.plot(t_loss, label=f'Fold {i + 1} - Training Loss')
        if v_loss is not None:
            plt.plot(v_loss, label=f'Fold {i + 1} - Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses for Each Fold')
    plt.savefig(f"./FIGS/{prefix}/kfold_trainning")


async def send_telegram_message(message, bot_token='6053416210:AAHg6dOl_eGMeQWiYnFimryxgVlI-15Ttto',
                                chat_id='629647931'):
    bot = telegram.Bot(token=bot_token)
    await bot.send_message(chat_id=chat_id, text=message)



