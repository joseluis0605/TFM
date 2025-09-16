import numpy as np
import torch # libreria principal de python
import torch.nn as nn # modulo para las redes neuronales
import torch.optim as optim # modulo para algoritmos de optimizacion en redes neuronales
import torch.utils.data as data # modulo para tratar con los datasets
import math # operaciones matematicas
import copy # para copiar objetos y estructuras

# En esta clase vamos a aplicar la posicion al embedding para que el transformer tenga en cuenta la posicion de cada palabra
class PositionalEncoding(nn.Module):
    # le pasamos como atributos la dimension del embedding y la secuencia máxima de palabras
    def __init__(self, dim_embedding, tam_secuencia_maximo):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(tam_secuencia_maximo, dim_embedding)
        position = torch.arange(0, tam_secuencia_maximo, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_embedding, 2).float() * -(math.log(10000.0) / dim_embedding))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # tiene la forma: [batch_size, longitud_secuencia, dim_embedding]
        return x + self.pe[:, :x.size(1)]