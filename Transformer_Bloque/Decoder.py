import numpy as np
import torch # libreria principal de python
import torch.nn as nn # modulo para las redes neuronales
import torch.optim as optim # modulo para algoritmos de optimizacion en redes neuronales
import torch.utils.data as data # modulo para tratar con los datasets
import math # operaciones matematicas
import copy # para copiar objetos y estructuras
from TFM.Transformer_Bloque.MultiHeadAttention import MultiHeadAttention
from TFM.Transformer_Bloque.Encoder import Encoder
from TFM.Transformer_Bloque.FeedForward import FeedForward

# En esta clase vamos a implementar el decoder, el cual nos va a permitir generar la salida
class Decoder(nn.Module):
    def __init__(self, dim_embedding, num_cabezas, dim_capas_ocultas, dropout):
        super(Decoder, self).__init__()
        # es muy similar al encoder, pero con pequeñas diferencias.
        # una de ellas es la mascara (aqui si se pone a true)

        # Atributos
        # almacenamos los bloques de autoatencion (primero) y  de atencion cruzada (segundo) que es la que une encoder y decoder
        self.capa_autoatencion = MultiHeadAttention(dim_embedding, num_cabezas)
        self.atencion_cruzada = MultiHeadAttention(dim_embedding, num_cabezas)
        # almacenamos el bloque de add_norm en donde la suma se hara luego manualmente
        self.add_norm1 = nn.LayerNorm(dim_embedding)
        self.add_norm2 = nn.LayerNorm(dim_embedding)
        self.add_norm3 = nn.LayerNorm(dim_embedding)
        # almacenamos el bloque de Feed Forward (red neuronal)
        self.feed_forward = FeedForward(dim_embedding, dim_capas_ocultas)
        # almacenamos el bloqueo
        self.dropout = nn.Dropout(dropout)

    def forward(self, entrada_decoder, salida_encoder, mascara_atencion_encoder, mascara_atencion_decoder):
        # para la implementacion del flujo de ejecucion del decoder, necesitamos como parametros de la funcion:
        # la entrada del decoder (lo que seria el resultado que deseamos predecir desplazado a la derecha)
        # la salida del encoder
        # la mascara de atencion del decoder (puesta a 0 todo lo futuro)
        # la mascara de atencion del encoder (en donde cada palabra puede ver todo lo del futuro y pasado)

        # primero vamos a aplicar el bloque de autoatencion
        resultado_autoatencion = self.capa_autoatencion(entrada_decoder, entrada_decoder, entrada_decoder, mascara_atencion_decoder)

        # aplicamos bloque de add & normalizacion
        resultado_add1 = entrada_decoder + self.dropout(resultado_autoatencion)
        resultado_norm1 = self.add_norm1(resultado_add1)

        # aplicamos bloque de atencion cruzada
        resultado_atencion_cruzada = self.atencion_cruzada(resultado_norm1, salida_encoder, salida_encoder, mascara_atencion_encoder)

        # aplicamos bloque de add & normalizacion
        resultado_add2 = resultado_norm1 + self.dropout(resultado_atencion_cruzada)
        resultado_norm2 = self.add_norm2(resultado_add2)

        # aplicamos bloque de Feed Forward
        resultado_feedforward = self.feed_forward(resultado_norm2)

        # aplicamos bloque de add & norm
        resultado_add3 = resultado_norm2 + self.dropout(resultado_feedforward)
        resultado_decoder = self.add_norm3(resultado_add3)

        return resultado_decoder