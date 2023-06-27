import pandas as pd 
import re
import string

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



df = pd.read_csv('/home/mriciba/Projects/dipsy/RoBERTa/data/ISEAR.csv', names=['Emotion', 'Text', 'DNTKNOW'])
df = df.drop(columns=['DNTKNOW'])
df = df.dropna()


y = df['Emotion']
x = df['Text']
# PREPROCESSING DATA
df['Text_processed'] = df.Text.apply(lambda x: clean_text(x))
x = df['Text_processed']
df_processed = df.drop(columns=['Text'])

# We check the length of the longest sentence
length = 0
umbral = 280
count = 0
for i in range(len(x)):
    if len(x[i]) > length:
        length = len(x[i])
    if len(x[i]) > umbral:
        count += 1

df_umbral = df_processed[df.Text_processed.str.len() > umbral]


print(
    f'Hacemos un análisis de la longitud de carácteres más grande ya que en el uso de transformers será necesario conocerla\n{length} Hay {count} frases que superan {umbral} caracteres')


