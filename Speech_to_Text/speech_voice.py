# importamos librerías
import numpy as np
import sounddevice as sd # librería que nos permite trabajar con el microfono
import whisper # modelo para pasar voz -> texto

# definimos el modelo (tamaño pequeño + device = cpu)
modelo = whisper.load_model("small", device="cpu")

# definimos funcion que capture y prepare el sonido con 16000 capturas por segundo
def speech_to_text(capturas_segundo: 16000):
    print("Capturando sonido...")
    lista_grabacion = []

    # definimos funcion que coja las capturas y las
    # almacene
    def callback(indata, frames, time, status):
        if status: # si hay algun error, lo muestre
            print(status)
        # añadimos bloque de frames a nuestra lista
        # usamos funcion copy ya que indata es una variable que se reutiliza y así nos evitamos sobreescrituras
        lista_grabacion.append(indata.copy())

    # nos protegemos de algun error durante la captura
    try:
        # abrimos el microfono hasta que se pulse enter
        # el microfono abierto captura rafagas de sonido y se las pasa a callback para juntarlas todas
        with sd.InputStream(callback=callback, samplerate=capturas_segundo, channels=1):
            print("Pulsa ENTER para cerrar el microfono")
            input()
    except Exception as e:
        print(e)
        return None, None

    # concatenamos todos los bloques de audio en uno solo
    audio_final = np.concatenate(lista_grabacion, axis=0) # tiene forma [numero de muestras, canales]
    audio_final = audio_final.squeeze(axis=1)  # quita solo el eje de los canales, quitamos una dimension
    valor_maximo_positivo = np.max(np.abs(audio_final)) # obtenemos el valor maximo en positivo para normalizar

    # el valor maximo puede ser 0 si no se habla, por lo que nos protegemos
    if valor_maximo_positivo > 0:

        audio_final = audio_final / valor_maximo_positivo
    else:
        audio_final = audio_final.astype("float32") # se coloca el mismo formato pero todo a cero

    # retornamos el audio final procesado y unido, y el numero de muestras por segundo
    return audio_final, capturas_segundo

# llamamos a la funcion
audio, sr = speech_to_text(capturas_segundo=16000)

# usamos el modelo para predecir el audio y generar texto.
# le pasamos el audio ya procesado y fp16=16 para que sea float32 y haya mejor precision
texto_generado = modelo.transcribe(audio, fp16=False)

# mostramos texto
print(texto_generado)
