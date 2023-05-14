from transformers import RobertaModel, AutoConfig, get_linear_schedule_with_warmup, AdamW, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from utils import *

# Check if GPU is available and set the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the dataset
df = load_and_preprocess_data('../data/isear.csv')

# Set max sequence length and load the tokenizer
MAX_LEN = 180
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
# Tokenize and encode the input data
tokenized_features = tokenize_and_encode(df, tokenizer, MAX_LEN)

# Encode the emotion labels using LabelEncoder
le = LabelEncoder()
target_num = le.fit_transform(df['Emotion'].values.tolist())

# Split the dataset into training and validation sets
train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = split_data(
    tokenized_features, target_num)

# Set the batch size for training
batch_size = 16

# Create DataLoaders for training and validation sets
train_dataloader = create_dataloader(train_inputs, train_masks, train_labels, batch_size)
validation_dataloader = create_dataloader(validation_inputs, validation_masks, validation_labels, batch_size)

# Cargar el modelo desde el directorio
model_path = "./RoBERTa_entrenado"
model = CustomRoBERTa.from_pretrained(model_path,7)
model.to(device)


# Descongelar solo las últimas dos capas de RoBERTa
for param in model.base_model.encoder.layer[-2:].parameters():
    param.requires_grad = True

# Definir el optimizador, el planificador de la tasa de aprendizaje y la función de pérdida
optimizer = AdamW(model.parameters(), lr=3e-6)
epochs = 100  # Modificar el número de épocas para el reentrenamiento según sea necesario
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_function = nn.CrossEntropyLoss()

# Reentrenar el modelo personalizado
retraining_losses, retraining_accuracies, revalidation_losses, revalidation_accuracies = train_model(
    model, train_dataloader, validation_dataloader, optimizer, scheduler, loss_function, device, epochs
)

# Guardar el modelo reentrenado y su configuración en un directorio
retrained_save_directory = "roberta_retrained"
retrained_classifier_state_dict_path = os.path.join(retrained_save_directory, "retrained_classifier_state_dict.pt")
torch.save(model.classifier.state_dict(), retrained_classifier_state_dict_path)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(retraining_losses, label="Retraining Loss")
plt.plot(revalidation_losses, label="Revalidation Loss")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(retraining_losses, label="Retraining Loss")
plt.plot(revalidation_losses, label="Revalidation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(retraining_accuracies, label="Retraining Accuracy")
plt.plot(revalidation_accuracies, label="Revalidation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("Model_retrained.png")
