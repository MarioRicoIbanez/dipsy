import re
import warnings
import os
import asyncio
import traceback
from utils import *
from transformers import AutoTokenizer, RobertaModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import mlflow


# Añadiendo las librerías necesarias para el encoding de los datos.


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#set mlflow environment


#create a for loop from 0 to -5

for i in range(0, -13, -1):

    """GLOBAL PARAMS"""
    MAX_LEN = 340
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    EPOCHS = 100
    #if i = 0 LEARNING_RATE = 3e-4 else LEARNING_RATE = 3-e-5
    if i == 0:
        LEARNING_RATE = 3e-3
    else:
        LEARNING_RATE = 3e-5
    K_FOLDS = 1
    BATCH_SIZE = 64
    DATASET_NAME = "RikoteMaster/Emotion_Recognition_4_llama2_chat"
    LOSS_FUNCTION = nn.CrossEntropyLoss()
    LAYERS_TO_UNFREEZE = i
    PATIENCE = 4

    TEST_DATASET = "RikoteMaster/isear_augmented"





    #log params as different values

    run_name = f"Entrenamiento roberta base con {LAYERS_TO_UNFREEZE} capas descongeladas, LR mixed"
    mlflow.set_tracking_uri("/workspace/NASFolder/mlruns")
    mlflow.set_experiment("RoBERTa_LAYERED_TRAINNING")
    mlflow.start_run(run_name=run_name)

    mlflow.log_param("MAX_LEN", MAX_LEN)
    mlflow.log_param("TEST_SIZE", TEST_SIZE)
    mlflow.log_param("RANDOM_STATE", RANDOM_STATE)
    mlflow.log_param("EPOCHS", EPOCHS)
    mlflow.log_param("LEARNING_RATE", LEARNING_RATE)
    mlflow.log_param("K_FOLDS", K_FOLDS)
    mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
    mlflow.log_param("DATASET_NAME", DATASET_NAME)
    mlflow.log_param("LAYERS_TO_UNFREEZE", LAYERS_TO_UNFREEZE)
    mlflow.log_param("PATIENCE", PATIENCE)
    #register the model with mlflow
    mlflow.pytorch.autolog(log_models=True)


    "DIRECTORIES"
    #now LOAD_DIRECTORY


    if LAYERS_TO_UNFREEZE == -1:
        LOAD_DIRECTORY = "/workspace/NASFolder/MODELS/RoBERTa_entrenado_base"
    elif LAYERS_TO_UNFREEZE != 0:
        LOAD_DIRECTORY = f"/workspace/NASFolder/MODELS/RoBERTa_entrenado_base_{LAYERS_TO_UNFREEZE+1}"

    if LAYERS_TO_UNFREEZE != 0: 
        SAVE_DIRECTORY = f"/workspace/NASFolder/MODELS/RoBERTa_entrenado_base_{LAYERS_TO_UNFREEZE}"
        classifier_state_dict_path_save = os.path.join(
            SAVE_DIRECTORY, "classifier_state_dict.pt")
        
    else:
        SAVE_DIRECTORY = f"/workspace/NASFolder/MODELS/RoBERTa_entrenado_base"
        classifier_state_dict_path_save = os.path.join(
            SAVE_DIRECTORY, "classifier_state_dict.pt")

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

   
    train_inputs = tokenized_features['input_ids']
    train_masks = tokenized_features['attention_mask']
    train_labels = target_one_hot


    if LAYERS_TO_UNFREEZE != 0:
        classifier_state_dict_path_load = os.path.join(
            LOAD_DIRECTORY, "classifier_state_dict.pt")
        classifier_state_dict = torch.load(classifier_state_dict_path_load)
        


    base_model = RobertaModel.from_pretrained("roberta-base")

    if LAYERS_TO_UNFREEZE == 0: 
        for param in base_model.parameters():
            param.requires_grad = False
        
    else: 
        for param in base_model.encoder.layer[:LAYERS_TO_UNFREEZE].parameters():
            param.requires_grad = True
            

        # Initialize the custom RoBERTa model
    model = CustomRoBERTa(base_model.to(device), num_classes=len(set(ds['Emotion'])))
    if LAYERS_TO_UNFREEZE != 0:
        model.classifier.load_state_dict(classifier_state_dict)


    compiled_model = torch.compile(model)

    optimizer = torch.optim.AdamW(compiled_model.parameters(), LEARNING_RATE)

    total_steps = len(train_inputs) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                                num_training_steps=total_steps)

    kfold_results, model = kfold_cross_validation(train_inputs, train_labels, train_masks, compiled_model, device, EPOCHS, lr=LEARNING_RATE,
                                                        k_folds=K_FOLDS,
                                                        batch_size=BATCH_SIZE, patience = PATIENCE)

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

    # Suponiendo que training_losses_frozen, training_accuracies_frozen,
    # validation_losses_frozen, y validation_accuracies_frozen son listas de listas

    def log_metrics_to_mlflow(metrics, metric_name):
        for i, metric_list in enumerate(metrics):
            for j, metric_value in enumerate(metric_list):
                mlflow.log_metric(f"{metric_name}", metric_value, step=i)

    # Llamando a la función para cada conjunto de métricas
    log_metrics_to_mlflow(training_losses_frozen, "training_loss_frozen")
    log_metrics_to_mlflow(training_accuracies_frozen, "training_accuracy_frozen")
    log_metrics_to_mlflow(validation_losses_frozen, "validation_loss_frozen")
    log_metrics_to_mlflow(validation_accuracies_frozen, "validation_accuracy_frozen")


                        
    print(training_losses_frozen, training_accuracies_frozen)






    model.base_model.save_pretrained(SAVE_DIRECTORY)

    torch.save(model.classifier.state_dict(), classifier_state_dict_path_save)
    print("Model saved successfully")

    del train_inputs, train_masks, train_labels, tokenized_features, target_one_hot, ds
    del base_model 
    del compiled_model
    torch.cuda.empty_cache()



    from sklearn.metrics import classification_report

    # Cargar el conjunto de datos de prueba
    test_ds = load_dataset(TEST_DATASET, split="test")

    # Procesar el conjunto de datos de prueba
    tokenized_test_features = tokenizer.batch_encode_plus(test_ds['Text_processed'], add_special_tokens=True,
                                                          padding='max_length', truncation=True,
                                                          max_length=MAX_LEN, return_attention_mask=True,
                                                          return_tensors='pt'
                                                          )

    test_inputs = tokenized_test_features['input_ids']
    test_masks = tokenized_test_features['attention_mask']
    test_labels = one_hot_encoder.transform(np.array(test_ds['Emotion']).reshape(-1, 1)).toarray()

    # Cargar el modelo entrenado

    


    # Crear un TensorDataset para los datos de prueba
    test_dataset = TensorDataset(test_inputs, test_masks, torch.tensor(np.argmax(test_labels, axis=1)))

    # Crear un DataLoader para los datos de prueba
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

       # Preparar el modelo para evaluar
    model.eval()

    all_preds = []
    all_label_ids = []

    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_labels = batch

            # Mover los lotes al dispositivo (asumiendo que 'device' está definido)
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)

            # Realizar la predicción
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits

            # Mover logits y etiquetas a CPU para su procesamiento
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Guardar predicciones y etiquetas
            all_preds.extend(np.argmax(logits, axis=1))
            all_label_ids.extend(label_ids)

    # Calcular el accuracy
    accuracy = accuracy_score(all_label_ids, all_preds)

    # Asegúrate de que test_labels_binary esté definido correctamente
    # Calcular las métricas de rendimiento
    report = classification_report(all_label_ids, all_preds, output_dict=True)

    # Registrar las métricas de rendimiento en MLflow
    mlflow.log_metric("test_precision", report["macro avg"]["precision"])
    mlflow.log_metric("test_recall", report["macro avg"]["recall"])
    mlflow.log_metric("test_f1-score", report["macro avg"]["f1-score"])
    mlflow.log_metric("test_accuracy", accuracy)

    print(report)



    
    # Al final de cada iteración del bucle for, libera la memoria de los tensores y modelos que ya no necesitas
    del test_inputs, test_masks, test_labels, tokenized_test_features, test_ds
    del optimizer, scheduler, kfold_results
    torch.cuda.empty_cache()

    mlflow.end_run()

