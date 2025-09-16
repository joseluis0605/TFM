import numpy as np
import torch  # libreria principal de python
import torch.nn as nn  # modulo para las redes neuronales
import torch.optim as optim  # modulo para algoritmos de optimizacion en redes neuronales
import torch.utils.data as data  # modulo para tratar con los datasets
import math  # operaciones matematicas
import copy  # para copiar objetos y estructuras
from TFM.Transformer_Bloque.MultiHeadAttention import MultiHeadAttention
from TFM.Transformer_Bloque.FeedForward import FeedForward

# En esta clase vamos a definir nuestro ENCODER, el cual se va a centrar en captar toda la información de la entrada, sacando
# sus relaciones mas profundas entre las distintas palabras

class Encoder(nn.Module):
    def __init__(self, dim_embedding, num_cabezas, dim_capas_ocultas, dropout):
        super(Encoder, self).__init__()
        # los parametros que les pasamos son los siguientes:
        # dim_embedding: es la dimensión que tienen los embeddings
        # num_cabezas: el numero de cabezas en las que se va a dividir el embedding
        # dim_capas_ocultas: numero de neuronas de la capa interna en la parte de Feed Forward
        # dropout: se empleara esta tecnica de bloqueo sobre los tensores de los datos de entrada para que se mejore el aprendizaje

        # Definimos los atributos
        self.multiHeadAttention = MultiHeadAttention(dim_embedding, num_cabezas) # definimos el bloque de atencion
        self.feedForward = FeedForward(dim_embedding, dim_capas_ocultas) # definimos el bloque de redes neuronales
        self.add_norm1 = nn.LayerNorm(dim_embedding) # bloque de normalizacion (el add se hace manual)
        self.add_norm2 = nn.LayerNorm(dim_embedding) # bloque de normalizacion (el add se hara manual)
        self.dropout = nn.Dropout(dropout)

    def forward(self, entrada, mask):
        # los datos que nos van a entrar va a ser un modelo formado por el embedding + posicional encoding
        # la entrada tiene la siguiente forma: [batch_size, longitud_secuencia, dim_embedding]
        # le pasamos la entrada al bloque de atencion
        resultado_atencion = self.multiHeadAttention(entrada, entrada, entrada, mask)

        # en el bloque de add & norm procedemos a sumar la entrada con el resultado de la atencion y aplicamos normalizacion
        # añadimos un dropout para bloquear ciertas partes del tensor y que asi aprenda mejor el modelo
        resultado_add = entrada + self.dropout(resultado_atencion)
        # aplicamos la normalizacion
        resultado_normalizacion = self.add_norm1(resultado_add)

        # aplicamos el bloque de Feed Forward (red neuronal)
        resultado_feedForward = self.feedForward(resultado_normalizacion)

        # aplicamos ultimo bloque de normalizacion y de add
        resultado_suma2 = resultado_normalizacion + self.dropout(resultado_feedForward)
        # aplicamos normalizacion
        resultado_encoder = self.add_norm2(resultado_suma2)

        # la forma que tendra es: [batch_size, tam_secuencia, dim_embedding]
        return resultado_encoder