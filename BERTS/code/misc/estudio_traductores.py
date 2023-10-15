import time
from deep_translator import (GoogleTranslator,
                             MyMemoryTranslator)

def translate_text(text):
    traductor = GoogleTranslator(source='es', target='en')
    resultado = traductor.translate(text)
    return resultado

a_traducir = "Solo queda saber como de bien traduce, las cosas un poco raras o incluso a vees errores tipográficos. "

def test_traductor(traductor):
    for i in range(50):
        inicio = time.time()
        translation = traductor.translate(a_traducir)
        fin = time.time()
    print(f"El traductor {traductor} tardó  de media {(fin-inicio)/50} en traducir 50 veces {a_traducir} a {translation}")

traductor = GoogleTranslator(source='es', target='en')
test_traductor(traductor)

traductor = MyMemoryTranslator(source='es', target='en')
test_traductor(traductor)




