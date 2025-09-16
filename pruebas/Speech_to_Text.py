import sounddevice as sd # libreria para abrir el microfono
import soundfile as sf # para pasar de voz -> texto
import whisper #
import numpy as np

# Cargar modelo Whisper en CPU
model = whisper.load_model("small", device="cpu")
print("Usando CPU para la transcripción")

# Función para grabar hasta ENTER
def grabar_audio_hasta_enter(samplerate=16000):
    print("Grabando... pulsa ENTER para detener la grabación")
    grabacion = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        grabacion.append(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback): #aqui es donde se abre el microfono
        input()  # espera a que pulses ENTER
        print("Grabación detenida")

    # Concatenar todos los bloques
    audio_final = np.concatenate(grabacion, axis=0)

    # Normalizar entre -1 y 1
    audio_final = audio_final / np.max(np.abs(audio_final))
    return audio_final, samplerate

# Grabar audio
audio, sr = grabar_audio_hasta_enter()

# Guardar temporalmente el audio en un archivo WAV
temp_file = "temp_audio.wav"
sf.write(temp_file, audio, sr)

# Transcribir con Whisper
result = model.transcribe(temp_file, fp16=False)
print("Transcripción:", result["text"])
