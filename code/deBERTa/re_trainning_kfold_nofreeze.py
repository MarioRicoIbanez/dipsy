import os
import re
import warnings
import asyncio
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from transformers import DebertaTokenizer, DebertaModel, DebertaConfig, get_linear_schedule_with_warmup

from utils import *

# Suppress certain warnings
TYPED_STORAGE_WARNING = re.compile(".*TypedStorage is deprecated.*")
FALLBACK_KERNEL_WARNING = re.compile(".*Using FallbackKernel: aten.cumsum.*")
TRITON_RANDOM_WARNING = re.compile(".*using triton random, expect difference from eager.*")


def warning_filter(message, category, filename, lineno, file=None, line=None):
    if category == UserWarning and (TYPED_STORAGE_WARNING.match(str(message)) or FALLBACK_KERNEL_WARNING.match(
            str(message)) or TRITON_RANDOM_WARNING.match(str(message))):
        return False
    return True


warnings.showwarning = warning_filter
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.ir")

# Set global parameters
MAX_LEN = 180
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 16
LOSS_FUNCTION = nn.CrossEntropyLoss()
EPOCHS_KFOLD_UNFROZEN = 5
LEARNING_RATE_KFOLD_UNFROZEN = 5e-5
K_FOLDS_UNFROZEN = 5
LAYERS_TO_UNFREEZE = -2

# Set directories for saving models
SAVE_DIRECTORY_FROZEN = "./MODELS/DeBERTa_entrenado_kfold"
if not os.path.exists(SAVE_DIRECTORY_FROZEN):
    os.makedirs(SAVE_DIRECTORY_FROZEN)
SAVE_DIRECTORY_UNFROZEN = f"./MODELS/DeBERTa_entrenado_kfold {LAYERS_TO_UNFREEZE} layers unfrozen"
if not os.path.exists(SAVE_DIRECTORY_UNFROZEN):
    os.makedirs(SAVE_DIRECTORY_UNFROZEN)

# Disable parallel tokenization
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    asyncio.run(
        send_telegram_message(
            message=f'Comenzando a entrenar el modelo DeBERTa con {LAYERS_TO_UNFREEZE} capas descongeladas'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    df = load_and_preprocess_data('../../data/isear.csv')

    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")

    tokenized_features = tokenizer.batch_encode_plus(
        df['Text_processed'].values.tolist(),
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_attention_mask=True,
        return_tensors='pt'
    )

    one_hot_encoder = OneHotEncoder()
    target_one_hot = one_hot_encoder.fit_transform(df['Emotion'].values.reshape(-1, 1)).toarray()
    classes = one_hot_encoder.categories_[0]

    train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks = train_test_split(
        tokenized_features["input_ids"],
        target_one_hot,
        tokenized_features["attention_mask"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target_one_hot,
    )

    classifier_state_dict_path = os.path.join(SAVE_DIRECTORY_FROZEN, "classifier_state_dict.pt")
    classifier_state_dict = torch.load(classifier_state_dict_path)

    configuration = DebertaConfig()
    base_model = DebertaModel(configuration)
    configuration = base_model.config

    for param in base_model.encoder.layer[LAYERS_TO_UNFREEZE:].parameters():
        param.requires_grad = True

    unfrozen_model = CustomDeBERTa(base_model.to(device), num_classes=len(set(df['Emotion'])))

    # Load the frozen classifier state dict into the unfrozen model

    unfrozen_model.classifier.load_state_dict(classifier_state_dict)
    # Set the optimizer and scheduler
    optimizer = optim.AdamW(unfrozen_model.parameters(), lr=LEARNING_RATE_KFOLD_UNFROZEN)
    total_steps = len(train_inputs) * EPOCHS_KFOLD_UNFROZEN
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                                num_training_steps=total_steps)

    # Perform k-fold cross-validation on the unfrozen model
    kfold_results_unfrozen, model_unfrozen_last_layer = kfold_cross_validation(train_inputs, train_labels, train_masks,
                                                                               unfrozen_model, device,
                                                                               EPOCHS_KFOLD_UNFROZEN,
                                                                               lr=LEARNING_RATE_KFOLD_UNFROZEN,
                                                                               k_folds=K_FOLDS_UNFROZEN,
                                                                               batch_size=BATCH_SIZE,
                                                                               patience=2)

    # Clear unused objects and free up memory
    del unfrozen_model
    torch.cuda.empty_cache()

    # Save the RoBERTa model and the classifier state dict
    model_unfrozen_last_layer.base_model.save_pretrained(SAVE_DIRECTORY_UNFROZEN)
    classifier_state_dict_path = os.path.join(SAVE_DIRECTORY_UNFROZEN, "classifier_state_dict.pt")
    # if not created the directory, create it

    torch.save(model_unfrozen_last_layer.classifier.state_dict(), classifier_state_dict_path)

    # Send results to Telegram
    for i, fold_result in enumerate(kfold_results_unfrozen[:-1]):
        message = f"Fold {i + 1} results (unfrozen model):\nAccuracy: {fold_result['validation_accuracies'][LAYERS_TO_UNFREEZE]:.4f}\nLoss: {fold_result['validation_losses'][LAYERS_TO_UNFREEZE]:.4f}"
        asyncio.run(send_telegram_message(message=message))
    test_dataloader = create_dataloader(torch.tensor(test_inputs), torch.tensor(test_masks), torch.tensor(test_labels),
                                        BATCH_SIZE, device)
    test_losses_unfrozen, test_accuracy_unfrozen, y_true_one_hot, y_pred_one_hot = test_model(model_unfrozen_last_layer,
                                                                                              test_dataloader,
                                                                                              LOSS_FUNCTION)

    plot_confusion_matrix_one_hot(np.array(y_true_one_hot), np.array(y_pred_one_hot), classes, LAYERS_TO_UNFREEZE)

    message = f"Unfrozen model test results:\nAccuracy: {test_accuracy_unfrozen:.4f}\nLoss: {test_losses_unfrozen:.4f}"
    asyncio.run(send_telegram_message(message=message))

    # Save and plot the results
    all_training_losses_unfrozen = [fold_result['training_losses'] for fold_result in kfold_results_unfrozen]
    all_training_accuracies_unfrozen = [fold_result['training_accuracies'] for fold_result in kfold_results_unfrozen]
    all_validation_losses_unfrozen = [fold_result['validation_losses'] for fold_result in kfold_results_unfrozen]
    all_validation_accuracies_unfrozen = [fold_result['validation_accuracies'] for fold_result in
                                          kfold_results_unfrozen]
    save_and_plot_results(all_training_losses_unfrozen, all_validation_losses_unfrozen,
                          f"Unfrozen {LAYERS_TO_UNFREEZE} layers")

except Exception as e:
    # Send error message to Telegram
    error_message = f"Error occurred: {str(e)}\n\n{traceback.format_exc()}"
    asyncio.run(send_telegram_message(message=error_message))
    # Print error message to console
    print(error_message)
