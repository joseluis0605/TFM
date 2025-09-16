import sounddevice as sd # libreria para utilizar el microfono
import whisper # modelo para hacer la conversion de voz -> texto
import numpy as np # libreria matematica de matrices y arrays

# Cargar modelo Whisper en CPU
# cargamos el modelo version small en la cpu
model = whisper.load_model("small", device="cpu")
print("Usando CPU para la transcripción")

# Función para grabar hasta ENTER
# samplerate= 16000: cantidad de muestras que coge por sonido
def grabar_audio_hasta_enter(samplerate=16000):
    print("Grabando... pulsa ENTER para detener la grabación")
    # va a almacenar los bloques concatenados
    # un bloque es un conjunto de muestras
    grabacion = []

    # cogera cada bloque y lo irá apilando a la grabacion
    # forma del bloque = [frames, canales_audio]
    # indata = bloque
    # frame = muestra
    # status: si hubo algun error durante la captura de sonido --> se imprime
    def callback(indata, frames, time, status):
        if status:
            print(status)
        # añadimos el bloque a la grabacion
        # indata es un buffer, que sounddevice reutiliza, por lo que debemos hacer copy para que no apunten al mismo lado siempre
        # grabacion es una lista de bloques (cada bloque tiene una muestra o frame)
        grabacion.append(indata.copy())

    # abre el microfono y llama muchas veces a la funcion callback para juntar los bloques
    # se ejecuta hasta que se pulsa ENTER
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        input()  # espera a que pulses ENTER
        print("Grabación detenida")

    # Concatenar todos los bloques (tenemos una lista de bloques)
    # la forma que tiene audio final es: [frames, 1]
    audio_final = np.concatenate(grabacion, axis=0)

    # Normalizar entre -1 y 1 (Whisper requiere float32), como en todo modelo
    audio_final = audio_final / np.max(np.abs(audio_final))
    return audio_final.astype(np.float32), samplerate

# Grabar audio
audio, sr = grabar_audio_hasta_enter()

#Pasar directamente el array de NumPy a Whisper
result = model.transcribe(audio, fp16=False)
print("Transcripción:", result["text"])
