from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, RobertaModel, get_linear_schedule_with_warmup
from utils import *
from datasets import load_dataset
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
K_FOLDS_FROZEN = 2
BATCH_SIZE = 16
LOSS_FUNCTION = nn.CrossEntropyLoss()

"DIRECTORIES"
SAVE_DIRECTORY_FROZEN = "./GLUE/RoBERTa_entrenado_kfold"
SAVE_DIRECTORY_UNFROZEN = "./GLUE/RoBERTa_entrenado_kfold_unfrozen_last_layer"

TYPED_STORAGE_WARNING = re.compile(".TypedStorage is deprecated.")
FALLBACK_KERNEL_WARNING = re.compile(".Using FallbackKernel: aten.cumsum.")
TRITON_RANDOM_WARNING = re.compile(".using triton random, expect difference from eager.")

def warning_filter(message, category, filename, lineno, file=None, line=None):
    if category == UserWarning and (TYPED_STORAGE_WARNING.match(str(message)) or FALLBACK_KERNEL_WARNING.match(str(message)) or TRITON_RANDOM_WARNING.match(str(message))):
        return False
    return True


warnings.showwarning = warning_filter
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.ir")



# Load the GLUE dataset
def load_glue_dataset(task_name):
    dataset = load_dataset('glue', task_name)
    return dataset

# Preprocess the GLUE dataset
def preprocess_glue_data(dataset, tokenizer, max_len):
    def encode(example):
        return tokenizer(example['sentence'], truncation=True, padding='max_length', max_length=max_len)

    encoded_dataset = dataset.map(encode, batched=True)
    return encoded_dataset



try:
    asyncio.run(send_telegram_message(message='Comenzando a entrenar GLUE'))
    # Check if GPU is available and set the default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    task_name = 'sst2'  # Choose the task name from the GLUE dataset
    glue_dataset = load_glue_dataset(task_name)
    train_dataset = glue_dataset['train']

    # Set max sequence length and load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # Tokenize and encode the input data
    encoded_train_dataset = preprocess_glue_data(train_dataset, tokenizer, MAX_LEN)


    labels = encoded_train_dataset['label']
    
    # Split the dataset into training and test sets
    train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks = train_test_split(
        encoded_train_dataset["input_ids"],
        labels,
        encoded_train_dataset["attention_mask"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # Initialize the base RoBERTa model
    base_model = RobertaModel.from_pretrained("roberta-base")

    # Freeze the base model layers to prevent their weights from being updated during training
    for param in base_model.parameters():
        param.requires_grad = False

    # Initialize the custom RoBERTa model
    model = CustomRoBERTa(base_model.to(device), num_classes=len(set(labels)))

    compiled_model = torch.compile(model)

    # Define the optimizer, learning rate scheduler, and loss function
    optimizer = torch.optim.AdamW(compiled_model.parameters(), LEARNING_RATE_KFOLD_FROZEN)

    # Train the custom RoBERTa model using k-fold cross validation
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)

    print(train_inputs.shape, train_labels.shape, train_masks.shape)    
    print(train_inputs[0], train_labels[0], train_masks[0])
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
