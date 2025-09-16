import numpy as np
import torch # libreria principal de python
import torch.nn as nn # modulo para las redes neuronales
import torch.optim as optim # modulo para algoritmos de optimizacion en redes neuronales
import torch.utils.data as data # modulo para tratar con los datasets
import math # operaciones matematicas
import copy # para copiar objetos y estructuras
from TFM.Transformer_Bloque.Encoder import Encoder
from TFM.Transformer_Bloque.Decoder import Decoder
from TFM.Transformer_Bloque.PositionalEncoding import PositionalEncoding

#import Encoder
#import Decoder
#import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, tam_vocabulario_encoder, tam_vocabulario_decoder, dim_embedding, num_cabezas, num_capas_EncDec, num_capas_ocultas, tam_max_secuencia, dropout):
        super(Transformer, self).__init__()

        # los parametros que les pasamos al modelo son:
        # tamaño del conjunto de palabras del encoder
        # tamaño del conjunto de palabras del decoder
        # dimension del embedding
        # numero de cabezas en las que se va a dividir el embedding en la capa de atencion
        # num_capas_EncDec se refiere a que esta arquitectura se va a ejecutar x veces, para capturar mas informacion
        # num_capas_ocultas se refiere al numero de capas del bloque de Feed Forward
        # tam_max_secuencia se refiere al tamaño maximo de la secuencia de palabras de entrada
        # el dropout se usa para bloquear ciertas dimensiones de los tensores y asi mejorar el aprendizaje

        # Atributos
        # capa para aplicar el embedding a las entradas
        self.embedding_encoder = nn.Embedding(tam_vocabulario_encoder, dim_embedding)
        self.embedding_decoder = nn.Embedding(tam_vocabulario_decoder, dim_embedding)
        # capa para aplicar el posiciona encoding a los embeddings de entradas del encoder y decoder
        self.positional_encoding = PositionalEncoding(dim_embedding, tam_max_secuencia)

        # creamos los bloques de encoder y decoder
        capas_Encoder_lista = []
        for _ in range(num_capas_EncDec):
            capa = Encoder(dim_embedding, num_cabezas, num_capas_ocultas, dropout)
            capas_Encoder_lista.append(capa)

        self.bloque_encoder = nn.ModuleList(capas_Encoder_lista)

        capas_Decoder_lista = []
        for _ in range(num_capas_EncDec):
            capa = Decoder(dim_embedding, num_cabezas, num_capas_ocultas, dropout)
            capas_Decoder_lista.append(capa)
        self.bloque_decoder = nn.ModuleList(capas_Decoder_lista)

        # capa lineal del final
        self.capa_lineal = nn.Linear(dim_embedding, tam_vocabulario_decoder)
        self.dropout = nn.Dropout(dropout)

    # funcion que va a generar las mascaras de atencion del codificador y decodificador
    def generate_mask(self, entrada_encoder, entrada_decoder):
        src_mask = (entrada_encoder != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (entrada_decoder != 0).unsqueeze(1).unsqueeze(3)
        seq_length = entrada_decoder.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    # implementamos la funcion del flujo de ejecucion del transformer
    # se le pasa la entrada del encoder y del decoder
    def forward(self, entrada_encoder, entrada_decoder):
        # generamos la mascara de atencion del encoder y decoder
        mascara_encoder, mascara_decoder = self.generate_mask(entrada_encoder, entrada_decoder)
        # aplicamos embedding a la entrada del encoder y decoder
        # le aplicamos la codificacion posicional
        # aplicamos dropout
        entrada_encoder_embedding = self.dropout(self.positional_encoding(self.embedding_encoder(entrada_encoder)))
        entrada_decoder_embedding = self.dropout(self.positional_encoding(self.embedding_decoder(entrada_decoder)))

        # como vamos a utilizar varias capas de encoder y decoder. La salida de una, va a ser la entrada de otra,
        # de esta manera se va a ir aplicando el resultado a la nueva capa del encoder y decoder. Así captura más informacin

        # para el encoder
        salida_encoder = entrada_encoder_embedding
        for capa_encoder in self.bloque_encoder:
            salida_encoder = capa_encoder(salida_encoder, mascara_encoder)

        # para el decoder
        salida_decoder = entrada_decoder_embedding
        for capa_decoder in self.bloque_decoder:
            salida_decoder = capa_decoder(salida_decoder, salida_encoder, mascara_encoder, mascara_decoder)

        # la salida del decoder la pasamos por una capa lineal
        salida = self.capa_lineal(salida_decoder)
        return salida