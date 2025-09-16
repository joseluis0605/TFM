import numpy as np
import torch # libreria principal de python
import torch.nn as nn # modulo para las redes neuronales
import torch.optim as optim # modulo para algoritmos de optimizacion en redes neuronales
import torch.utils.data as data # modulo para tratar con los datasets
import math # operaciones matematicas
import copy # para copiar objetos y estructuras

from torch.onnx.symbolic_opset9 import contiguous


# aqui desarrollaremos el bloque de la capa de atencion
class MultiHeadAttention (nn.Module):

    # init: va a contener los atributos
    def __init__(self, dimension_embedding, num_cabezas):
        super(MultiHeadAttention, self).__init__() #heredamos de la clase padre
        # Como tenemos que dividir el embedding en cabezas, nos aseguramos que dim_embedding % num_cabezas sea 0
        # aplicamos assert para lanzar mensaje de error si no se cumple la condicion
        assert dimension_embedding % num_cabezas == 0, "dimension_embedding must be divisible by num_cabezas"

        # DEFINIMOS LOS ATRIBUTOS

        # inicializamos las dimensiones
        self.dimension_embedding = dimension_embedding # almacenamos la dimension del embedding
        self.num_cabezas = num_cabezas # almacenamos las cabezas en las que se dividiran los embeddings
        self.dimension_cabezas = dimension_embedding // num_cabezas # dimension de cada cabeza

        # definimos las capas lineales de V, K, Q
        # capa lineal con los pesos, que recibe de entrada el tamaño del embedding y saca como salida el tamaño del embedding
        self.pesos_V = nn.Linear(dimension_embedding, dimension_embedding) # simplemente es un Σ(xi · wi)
        self.pesos_Q = nn.Linear(dimension_embedding, dimension_embedding)
        self.pesos_K = nn.Linear(dimension_embedding, dimension_embedding)
        # cuando se hagan operaciones entre ellos, tenemos que pasarlos por una capa lineal mas
        self.pesos_Salida = nn.Linear(dimension_embedding, dimension_embedding)

    # esta funcion va a calcular la atencion entre las palabras, la relacion de cada una con el resto
    # Q, K, V son los vectores resultados de aplicar la capa lineal a cada uno
    # mask es si se aplica mascara o no (solo en el DECODER)
    def scaled_dot_product_attention (self, Q, K, V, mask = None):
        # Para calcular la atención se realizan los siguientes pasos:
        # 1. se dividen los vectores en cabezas
        # 2. se multiplican los vectores Q·K y obtenemos los puntajes y se dividen entre la raiz cuadrada de la dimensión del embedding
        # 3. se le aplica una funcion Softmax al resultado
        # 4. se multiplica el resultado por el vector V:  R·V

        # multiplicamos los tensores Q·K y dividimos entre la raiz de la dimension de cabezas
        # Q es un tensor de 4 dimensiones [batch_size, num_cabezas, tam_secuencia_palabras, dim_cabeza]
        # K es un tensor de 4 dimensiones [batch_size, num_cabezas, tam_secuencia_palabras, dim_cabeza]
        # V es un tensor de 4 dimensiones [batch_size, num_cabezas, tam_secuencia_palabras, dim_cabeza]
        # Tenemos que hacer una multiplicacion de Q por la traspuesta de K, debido que no se puede hacer 4x3 · 4x3
        # hacemos 4x3 · 3x4 = 4x4 (esta es la razon por la que cambiamos la dimensión)
        attention_valores = torch.matmul(Q, K.transpose(2, 3)) / np.sqrt(self.dimension_cabezas) # tensor: [batch_size, num_cabezas, tam_secuencia, dim_cabeza]

        # aplicamos la mascara en caso de que sea el DECODER
        if mask is not None:
            # rellena los 0 con valor -infinito (-1e9) para que se tengan en cuenta en las redes neuronales
            attention_valores = attention_valores.masked_fill(mask == 0, -1e9)

        # ahora aplicamos la funcion Softmax al resultado obtenido (obtenemos las probabilidades -> una lista con valores entre 0 y 1)
        # se le aplica a la ultima dimension
        # la forma que tendría ahora es: [batch_size, num_cabezas, dimension_sec_Q, dimension_sec_K]
        attention_valores_probabilidades = torch.softmax(attention_valores, dim=-1)

        # multiplicamos por el tensor V
        # [batch_size, num_heads, seq_len_q, seq_len_k] x [batch_size, num_heads, seq_len_v, dim_v]
        resultado = torch.matmul(attention_valores_probabilidades, V)

        # Forma del tensor: (batch_size, num_cabezas, secuencia_frase, dimension_cabeza)
        return resultado

    # esta funcion recibe el embedding y lo divide en cabezas
    # la dimension de la entrada es: [batch_size, num_secuencia_frase, dim_embedding]
    def division_cabezas (self, entrada):
        # obtenemos las dimensiones del tensor de entrada
        batch_size, secuencia_frase, dim_embedding = entrada.size()
        # la funcion view nos permite redimensionar un tensor. Luego cambiamos la posicion 1 por la 2
        # La estructura del tensor: [batch_size, num_cabezas,  secuencia_frase, dimension_cabeza]
        return entrada.view(batch_size, secuencia_frase, self.num_cabezas, self.dimension_cabezas).transpose(1, 2)

    # esta funcion va a unificar las cabezas creadas anteriormente
    def unir_cabezas(self, entrada):
        # hacemos el paso opuesto a la funcion de division de cabezas
        # Tensor de entrada: [batch_size, num_cabezas, secuencia_frase, dimension_cabeza]
        # Tensor de salida: [batch_size, secuencia_frase, dimension_cabeza]
        batch_size, _, sencuencia_palabra, dim_embedding = entrada.size()
        # La estructura del tensor: [batch_size, cabezas,  secuencia_frase, dimension_cabeza]
        return entrada.transpose(1, 2).contiguous().view(batch_size, sencuencia_palabra, self.dimension_embedding)

    # implementamos funcion forward en donde veremos la secuencia de ejecucion de la capa multiheadattention
    # le pasamos los tensores, Q, K, V y la mascara (para el decoder)
    def forward(self, Q, K, V, mask = None):
        # Pasos a seguir:
        # 1. pasamos los tensores Q, K, V por las capas lineales y dividimos en cabezas
        # 2. Q, K los pasamos por el calculo de la atencion
        # 3. concatenamos las cabezas
        # 4. pasamos por capa lineal

        # pasamos tensores por capas lineales
        resultado_Q = self.pesos_Q(Q)
        resultado_K = self.pesos_K(K)
        resultado_V = self.pesos_V(V)

        # dividimos en cabezas
        resultado_Q = self.division_cabezas (resultado_Q)
        resultado_K = self.division_cabezas (resultado_K)
        resultado_V = self.division_cabezas (resultado_V)

        # calculamos la atencion
        resultados_atencion = self.scaled_dot_product_attention(resultado_Q, resultado_K, resultado_V, mask)

        # combinamos las cabezas
        cabezas_unidas = self.unir_cabezas (resultados_atencion)

        # aplicamos capa lineal
        resultado_atencion_multihead = self.pesos_Salida(cabezas_unidas)

        return resultado_atencion_multihead