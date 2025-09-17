from gtts import gTTS
from playsound import playsound
from langdetect import detect
import tempfile
import os


def hablar(texto):
    # Detectar idioma automáticamente
    idioma_detectado = detect(texto)
    lang = 'es' if idioma_detectado == 'es' else 'en'

    # Crear objeto TTS
    tts = gTTS(text=texto, lang=lang)

    # Guardar en archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        archivo_temp = tmp_file.name
        tts.save(archivo_temp)

    # Reproducir
    playsound(archivo_temp)

    # Borrar archivo temporal
    os.remove(archivo_temp)


# Ejemplo de uso
hablar("Hola, ¿cómo estás?")
hablar("Artificial intelligence has rapidly transformed the way we interact with technology, creating new opportunities and challenges for society. From self-driving cars to virtual assistants, AI systems are increasingly integrated into our daily lives. These systems rely on vast amounts of data and sophisticated algorithms to make predictions, recognize patterns, and even generate creative content. As AI continues to evolve, ethical considerations become more critical, including concerns about privacy, bias, and accountability. Researchers and developers must ensure that AI technologies are designed responsibly and transparently, promoting fairness and accessibility while minimizing potential harm. By balancing innovation with ethical responsibility, we can harness the full potential of artificial intelligence to improve education, healthcare, transportation, and countless other fields.")

