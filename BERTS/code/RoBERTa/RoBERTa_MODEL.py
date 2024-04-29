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
MAX_LEN = 180
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS_KFOLD_FROZEN = 100
LEARNING_RATE_KFOLD_FROZEN = 3e-4
K_FOLDS_FROZEN = 5
BATCH_SIZE = 8
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

try:
    asyncio.run(send_telegram_message(message='Comenzando a entrenar'))
    # Check if GPU is available and set the default device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #Download dataset
    
    #if it is not already downloaded
    if not os.path.exists('/ISEAR.csv'):
        os.system('wget https://raw.githubusercontent.com/PoorvaRane/Emotion-Detector/master/ISEAR.csv -P ./')

    # Load and preprocess the dataset
    df = load_and_preprocess_data('/home/mriciba/Projects/dipsy/RoBERTa/data/ISEAR.csv')
    df['Emotion'] = df['Emotion'].replace('guit', 'guilt')

    # Set max sequence length and load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # Tokenize and encode the input data
    tokenized_features = tokenizer.batch_encode_plus(
        df['Text_processed'].values.tolist(),
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Encode the emotion labels using LabelEncoder
    one_hot_encoder = OneHotEncoder()
    target_one_hot = one_hot_encoder.fit_transform(df['Emotion'].values.reshape(-1, 1)).toarray()

    # Split the dataset into training and test sets
    train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks = train_test_split(
        tokenized_features["input_ids"],
        target_one_hot,
        tokenized_features["attention_mask"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target_one_hot,
    )

    # Initialize the base RoBERTa model
    base_model = RobertaModel.from_pretrained("roberta-base")

    # Freeze the base model layers to prevent their weights from being updated during training
    for param in base_model.parameters():
        param.requires_grad = False

    # Initialize the custom RoBERTa model
    model = CustomRoBERTa(base_model.to(device), num_classes=len(set(df['Emotion'])))

    compiled_model = torch.compile(model)

    # Define the optimizer, learning rate scheduler, and loss function
    optimizer = torch.optim.AdamW(compiled_model.parameters(), LEARNING_RATE_KFOLD_FROZEN)

    total_steps = len(train_inputs) * EPOCHS_KFOLD_FROZEN
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                               num_training_steps=total_steps)

    # Train the custom RoBERTa model using k-fold cross validation

    kfold_results, model_frozen = kfold_cross_validation(train_inputs, train_labels, train_masks, compiled_model, device, EPOCHS_KFOLD_FROZEN, lr=LEARNING_RATE_KFOLD_FROZEN,
                                                        k_folds=K_FOLDS_FROZEN,
                                                        batch_size=BATCH_SIZE)

    # Iterate through the results of each fold except the last one
    for i, fold_result in enumerate(kfold_results[:-1]):
        message = f"Fold {i + 1} results (frozen model):\nAccuracy: {fold_result['validation_accuracies'][-1]:.4f}\nLoss: {fold_result['validation_losses'][-1]:.4f}"
        asyncio.run(send_telegram_message(message=message))

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

    # Save the base model and its configuration in a directory
    model_frozen.base_model.save_pretrained(SAVE_DIRECTORY_FROZEN)

    # Save the custom classifier's state_dict
    classifier_state_dict_path = os.path.join(SAVE_DIRECTORY_FROZEN, "classifier_state_dict.pt")
    torch.save(model_frozen.classifier.state_dict(), classifier_state_dict_path)

    # Create DataLoaders for the test set
    test_dataloader = create_dataloader(torch.tensor(test_inputs), torch.tensor(test_masks), torch.tensor(test_labels),
                                        BATCH_SIZE, device)

    # Evaluate the final model on the test set
    test_losses_frozen, test_accuracy_frozen, _, _ = test_model(model_frozen, test_dataloader, LOSS_FUNCTION)

    message = f"Frozen model test results:\nAccuracy: {test_accuracy_frozen:.4f}\nLoss: {test_losses_frozen:.4f}"
    asyncio.run(send_telegram_message(message=message))

    # Save and plot the results
    save_and_plot_results(all_training_losses_frozen, all_validation_losses_frozen, "frozen")

    del model_frozen
    torch.cuda.empty_cache()
except Exception as e:
    # Send error message to Telegram
    error_message = f"Error occurred: {str(e)}\n\n{traceback.format_exc()}"
    asyncio.run(send_telegram_message(message=error_message))
    # Print error message to console
    print(error_message)
