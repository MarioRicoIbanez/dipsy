from utils import *

df = pd.read_csv('../data/isear.csv', names=['Emotion', 'Text', 'DNTKNOW'])
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


# Separate the data in x_train, x_test, y_train, y_test

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)