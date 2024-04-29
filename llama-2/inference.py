import re, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from peft import PeftModel
import numpy as np 
import random



from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging
)



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



model_name = "meta-llama/Llama-2-7b-chat-hf"
adapters_name = "RikoteMaster/Llama-2-7b-chat-hf-Emotion_Recognition_4_llama2_chat"

print(f"Starting to load the model {model_name} into memory")



################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#################################


bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
    bnb_4bit_use_double_quant=use_nested_quant,
)

m = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0}
)

m.config.use_cache = False
m.config.pretraining_tp = 1



# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


model = PeftModel.from_pretrained(m, adapters_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


from datasets import load_dataset

ds = load_dataset("RikoteMaster/isear_rauw")

texts = ds['test']['Text_processed']
labels = ds['test']['Emotion']




# Initialize counters and lists for tracking
corrects = 0
wrong_detection = []
label_detection = []
exceptions = 0
predicted_labels = [] 
explainning = False

for sentence, label in zip(texts, labels):
    if not explainning:
        text = f"""<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows: Anger, Joy, Sadnes, Guilt, Shame, fear or disgust Sentence: {sentence} [/INST] """
    else:
        text = f"""<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows: Anger, Joy, Sadness, Guilt, Shame, Fear, or Disgust. Firstly, you have to express the explanation of why you think it's one emotion or another to make a pre-explanation. After that, you will predict the emotion expressed by the sentence. The format will be Explanation: and later Emotion: Sentence: {sentence} [/INST] """
    
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=len(tokenizer.tokenize(text)) + 5 if not explainning else len(tokenizer.tokenize(text)) + 70)
    result = pipe(text)
    print(result)
    try:
        pattern = r'\b(anger|joy|sadness|guilt|shame|fear|disgust)\b'

        # Assume we are in the context where classification with explanation is not required
        if not explainning:
            # Extract the first word after '[/INST]' marker, handling both upper and lower cases
            result_splitted = result[0]['generated_text'].split('[/INST]')[1]
            print(result_splitted)
            detected = re.search(pattern, result_splitted, flags=re.IGNORECASE)
            print(detected)
            detected = detected.group(0) if detected else 'None'
            print(detected)
            detected = detected.lower()
            predicted_labels.append(detected)
        else:
            # For classifying and explaining, extract the emotion after 'Emotion: '
            detected = 'None'
            result = result[0]['generated_text']
            result_splitted = result.split("Emotion: ")[2]
            match = re.search(pattern, result_splitted, flags=re.IGNORECASE)
            detected = match.group(0).lower() if match else 'None'  # Capitalize the detected emotion for consistency
            detected = detected.lower()
            predicted_labels.append(detected)

        
        print(detected)
        if label != detected:
            wrong_detection.append(str(result) + " THE TRUE LABEL IS "+ label)
            label_detection.append(detected)
        else:
            corrects += 1
    except:
        
        wrong_detection.append(str(result) + " THE TRUE LABEL IS " + label)
        label_detection.append(detected)
        exceptions += 1 
        
# Ensure that 'None' is included in both the actual and predicted labels
labels_extended = labels + ['none']  # This assumes 'labels' is a list of actual labels
labels_example = ["anger", "joy", "sadness", "guilt", "shame", "fear", "disgust", "none"]  # Include 'none' as a valid label

cm = confusion_matrix(labels, predicted_labels, labels=labels_example)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_example)
disp.plot(cmap=plt.cm.Blues)

# Define the directory
dir_name = "confusion_matrix/"

# Check if the directory exists
if not os.path.exists(dir_name):
    # If not, create the directory
    os.makedirs(dir_name)

# Now you can save the figure
plt.savefig(dir_name + 'confusion_matrix.png')  # Corrected file path to a writable directory

print("Confusion matrix saved as 'confusion_matrix.png'.")
print(f"Accuracy: {corrects / len(texts)}")