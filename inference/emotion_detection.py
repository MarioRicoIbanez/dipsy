
import os
os.system('pip install huggingface_hub transformers accelerate bitsandbytes peft deep_translator')

os.system('huggingface-cli login --token ')

import json
import transformers
import torch
from deep_translator import GoogleTranslator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_configuration(config_file):
    with open(config_file, 'r') as config_file:
        return json.load(config_file)

def load_model(model_id, peft_model_id, load_in_8bit, device_map):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, load_in_8bit=load_in_8bit)
    model.load_adapter(peft_model_id)
    return model

def translate_text(text, source_lang='es', target_lang='en'):
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    return translator.translate(text)

def classify_emotion(text, model, tokenizer):
    translated_text = translate_text(text)
    prompt = f"<s>[INST] In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. The emotions to consider are as follows: Anger, Joy, Sadness, Guilt, Shame, Fear, or Disgust Sentence: {translated_text} [/INST]"
    
    text_generation_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    sequences = text_generation_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=len(tokenizer.tokenize(prompt)) + 5
    )

    emotion = sequences[0]['generated_text'].split('[/INST]')[1].split()[0]
    return translated_text, prompt, emotion

if __name__ == "__main__":
    config = load_configuration('config.json')
    model_id = config["model_id"]
    peft_model_id = config["peft_model_id"]
    load_in_8bit = config["load_in_8bit"]
    device_map = config["device_map"]

    model = load_model(model_id, peft_model_id, load_in_8bit, device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Leer el JSON con el texto a clasificar
    with open('input.json', 'r') as input_file:
        input_data = json.load(input_file)
        input_text = input_data.get("text", "")

    

    if input_text:
        translated_text, prompt, emotion = classify_emotion(input_text, model, tokenizer)


    result = {
        "original_text": input_text,
        "translated_text": translated_text,
        "prompt_text": prompt,
        "emotion": emotion
    }

    # Guardar el resultado en un archivo JSON de salida
    with open('output.json', 'w') as output_file:
        json.dump(result, output_file, indent=4)

