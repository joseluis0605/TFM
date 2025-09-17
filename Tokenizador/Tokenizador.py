from langdetect import detect
from transformers import MBart50TokenizerFast
import torch

class TokenizadorBatch:
    def __init__(self, max_length=128, batch_size=1024, device=None):
        # Cargar el tokenizador una sola vez
        self.modelo_tokenizador = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt"
        )
        self.max_length = max_length # tamaño maximo de la secuencia
        self.batch_size = batch_size # tamaño del batch
        self.device = device if device else torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")

    def detector_idioma(self, texto):
        idioma = detect(texto)
        return "es_XX" if idioma == "es" else "en_XX"

    def tokenizar_fila(self, texto_encoder, texto_decoder):
        """
        Tokeniza una fila del dataset:
        - texto_encoder: sin tokens especiales
        - texto_decoder: con tokens especiales
        """
        # Detectamos idioma del encoder
        idioma = self.detector_idioma(texto_encoder)
        self.modelo_tokenizador.src_lang = idioma

        # Tokenizamos encoder (sin tokens especiales)
        tokens_encoder = self.modelo_tokenizador(
            texto_encoder,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).input_ids.to(self.device)

        # Tokenizamos decoder (con tokens especiales)
        tokens_decoder = self.modelo_tokenizador(
            texto_decoder,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).input_ids.to(self.device)

        return tokens_encoder, tokens_decoder

    def tokenizar_dataframe(self, df, columna_encoder="idioma", columna_decoder="traduccion"):
        """
        Tokeniza un DataFrame completo y añade columnas 'entrada_encoder' y 'entrada_decoder'.
        """
        encoder_tokens = []
        decoder_tokens = []

        for enc_texto, dec_texto in zip(df[columna_encoder], df[columna_decoder]):
            enc_tokens, dec_tokens = self.tokenizar_fila(enc_texto, dec_texto)
            encoder_tokens.append(enc_tokens)
            decoder_tokens.append(dec_tokens)

        df["entrada_encoder"] = encoder_tokens
        df["entrada_decoder"] = decoder_tokens
        return df
