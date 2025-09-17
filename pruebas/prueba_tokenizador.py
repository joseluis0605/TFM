from langdetect import detect
from transformers import MBart50TokenizerFast
import pandas as pd
import torch


class TokenizadorBatch:
    def __init__(self, max_length=128, batch_size=1024, device=None):
        # Cargar el tokenizador una sola vez
        self.modelo_tokenizador = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt"
        )
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device if device else torch.device("mps") if torch.has_mps else torch.device("cpu")

    def detector_idioma(self, texto):
        idioma = detect(texto)
        return "es_XX" if idioma == "es" else "en_XX"

    def tokenizar_batch(self, lista_textos):
        """
        Tokeniza un listado de frases en batches.
        Devuelve dos listas: tokens_encoder y tokens_decoder
        """
        tokens_encoder = []
        tokens_decoder = []

        # Procesamos por batch
        for i in range(0, len(lista_textos), self.batch_size):
            batch = lista_textos[i:i + self.batch_size]

            # Detectamos idioma de cada frase (puedes optimizar detectando idioma una sola vez si sabes que todo es español o inglés)
            idiomas = [self.detector_idioma(t) for t in batch]

            # Tokenización por batch
            # Nota: MBart no permite batch con distintos src_lang, por eso se hace por frase si hay idiomas mixtos
            for texto, idioma in zip(batch, idiomas):
                self.modelo_tokenizador.src_lang = idioma
                tokens_con = self.modelo_tokenizador(
                    texto,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                tokens_sin = self.modelo_tokenizador(
                    texto,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                # Movemos tensores a device
                tokens_encoder.append(tokens_sin.input_ids.to(self.device))  # encoder: sin tokens especiales
                tokens_decoder.append(tokens_con.input_ids.to(self.device))  # decoder: con tokens especiales

        return tokens_encoder, tokens_decoder

    def tokenizar_dataframe(self, df, columna_texto):
        """
        Tokeniza un DataFrame completo y añade columnas 'entrada_encoder' y 'entrada_decoder'
        """
        lista_textos = df[columna_texto].tolist()
        tokens_encoder, tokens_decoder = self.tokenizar_batch(lista_textos)

        # Guardamos en el DataFrame
        df["entrada_encoder"] = tokens_encoder
        df["entrada_decoder"] = tokens_decoder
        return df



