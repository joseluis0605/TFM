import numpy as np
import torch # libreria principal de python
import torch.nn as nn # modulo para las redes neuronales
import torch.optim as optim # modulo para algoritmos de optimizacion en redes neuronales
import torch.utils.data as data # modulo para tratar con los datasets
import math # operaciones matematicas
import copy # para copiar objetos y estructuras

# aqui vamos a crear el bloque de red neuronal
class FeedForward(nn.Module):
    # le pasamos la dimensión del embedding y la dimension de las capas internas
    def __init__(self, dim_embedding, dim_capas_ocultas):
        super(FeedForward, self).__init__() # heredamos de la clase padre
        # el transformer original tiene 2 capas y entre medio una funcion de activacion Relu
        self.capa_1 = nn.Linear(dim_embedding, dim_capas_ocultas)
        self.capa_2 = nn.Linear(dim_capas_ocultas, dim_embedding)
        self.funcion_activacion = nn.ReLU()

    # no aplicamos funcion de activacion a la capa de salida ya que posteriormente debemos de sumarlo con la propia entrada y aplicarle normalizacion
    # de esta manera no se perdería información de por medio
    def forward(self, entrada):
        resultado_1 = self.funcion_activacion(self.capa_1(entrada))
        resultado_2 = self.capa_2(resultado_1)
        return resultado_2