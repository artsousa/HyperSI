import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

import os
import spectral as sp
from source.sample import Sample


class HsiRoutine:

    def plot_mean_spectre(self, samples: list, **kwargs):
      
       """
          Plote o espectro médio, calculado a partir da matriz 2D, o eixo padrão são as linhas
          amostras: lista de matriz 2D (x*y, z)
          
          Parâmetros: 
              - samples: Lista de amostras a serem plotadas. 
              - Argumentos para plotagem do grafico (cor, espessura, etc. )
          Retorno
              - Imagem com o plot dos espectros médios
        """
        plot = []
        for matrix in samples:
            plot.append(plt.plot(np.arange(matrix.shape[1]),
                                 self.mean_from_2d(matrix=matrix), **kwargs)[0])

        return plot

    @staticmethod
    def get_wavelength(folder: str, sample: str, spectral_range=(1, 241)):
        
        """
            Função com objetivo de obter o valor do comprimento de onda, através dos arquivos
            gerados pela câmera hiperespectral, de cada amostra. 
            
            Parâmetros: 
                - folder: Caminho onde se encontra os arquivos
                - sample: Nome de cada amostra obtida
                - spectral_image: Tamanho do range espectral de cada amostra.
                
            Retorno:
                - Array com o tamanho do range espectral.
        """
        
        wv = sp.open_image(os.path.join(folder,
                                        sample,
                                        'capture', sample + '.hdr')).metadata['Wavelength']

        return np.array(wv)[spectral_range[0]:spectral_range[1]].reshape(1, -1)

    def raw2mat(self, image: Sample, white: Sample, dark: Sample,
                inplace=True):
         """
            Normaliza a amostra usando o Dark (0% Reflectance) e
            Referências brancas (100% de refletância), usando a equação:
            -log10((S - D)/(W - D))
            
            Parâmetros: 
                - image: A própria amostra sample, arquivo .hdr
                - white: Referência do branco, arquivo WHITEREF
                - dark: Referência do preto, arquivo DARKREF
                - inplace: significa altera o comportamento padrão, de modo que a operação no 
                dataframe não retorna nada, em vez disso, 'modifica os dados subjacentes'
                
            Funções: 
                extract_lines: Retorna somente as linhas obtidas na matrix
                    - Parâmetros: A própria matriz e a quantidade de linhas. 
                    - Retorno: Quantidade de linhas na matriz
                replace_median: Retorna matriz com a nova média substituída (verificando os locais na matriz em que são 
                                zero e substituindo pela próxima posição. 
                    - Parâmetros: A própria matriz
                    - Retorno: A matriz com valores substituidos da média
            Retorno: 
                - Matriz normalizada
        """

        def extract_lines(matrix, lines):
            rows = matrix.shape[1]
            return matrix[:, np.arange(0, rows, np.ceil(rows / lines)).astype(np.int), :]

        def replace_median(matrix):
            [_, rows, cols] = matrix.shape
            for z, x, y in zip(*np.where(matrix == 0)):
                if 0 < x < rows and 0 < y < cols:
                    window = matrix[z, x - 1:x + 2, y - 1:y + 2]

                    if len(np.where(window == 0)[0]) == 1:
                        matrix[z, x, y] = np.median(window[(window != 0)])

            return matrix

        extracted_dark = extract_lines(dark.sample, 25)
        extracted_white = extract_lines(white.sample, 25)

        raw_dark = self.mean_from_3d(matrix=extracted_dark, ndims=3, axis=1)
        raw_white = self.mean_from_3d(matrix=extracted_white, ndims=3, axis=1)
        raw_image = image.sample

        with np.errstate(divide='ignore', invalid='ignore'):
            pabs = np.nan_to_num(((raw_image - raw_dark) / (raw_white - raw_dark)), nan=0.0)

        normalized = replace_median(-np.log10((pabs * (pabs > 0)), where=(pabs > 0)))

        if inplace:
            image.normalized = normalized
            return

        return normalized

    @staticmethod
    def hsi2matrix(matrix: np.ndarray):
        """ 
            Reorganizar a matriz 3D para que cada pixel se torne um
            linha na matriz retornada 2D
            
            Parâmetros: 
                - matriz: Hipercubo em formato numpy
            Retorno: 
                - Matriz convertida em bidimensional
        """
        return matrix.T.reshape((matrix.shape[1] * matrix.shape[2], matrix.shape[0]), order='F')

    @staticmethod
    def matrix2hsi(matrix: np.ndarray, rows: int, cols: int):
      
        """           
            Reorganizar a matriz 2D em uma matriz 3D
            forma final (-1, linhas, colunas)
            
            Parâmetros: 
                - matriz: Matriz em formato numpy
                - rows: Número de linhas
                - cols: Número de colunas
            Retorno:
                - Matriz 3D
        """
        return matrix.T.reshape(-1, rows, cols)

    @staticmethod
    def mean_from_2d(matrix: np.ndarray, axis=0):
        """
            Retorna o espectro médio da matriz 2D
            matriz: hipercubo (x*y, z)
            
            Paramêtros: 
                - matriz: Matriz em formato numpy
                - axis: O eixo da matriz x ao longo do qual o filtro deve ser aplicado
            Retorno: 
                - Matriz com valor da média em 2 dimensões
        """
        return np.mean(matrix, axis=axis)

    @staticmethod
    def mean_from_3d(matrix: np.ndarray, ndims=2, axis=1):
        """
            Retornar o espectro médio da amostra 3D
            matriz: hipercubo (x, y, z)
            
            Parâmetros: 
                - matriz: A matriz em formato numpy
                - ndims: Número de dimensões
                - axis: O eixo da matriz x ao longo do qual o filtro deve ser aplicado
            Retorno
                - A média da matriz 3D
        """
        mean = np.mean(matrix, axis=axis).astype(np.float64)
        if ndims == 3:
            return mean.reshape((mean.shape[0], 1, mean.shape[1]))

        return mean

    @staticmethod
    def snv(matrix: np.ndarray):
          """
            Standard Normal Variate (SNV). 
            
            Parâmetros: 
                - matrix: Matriz em formato numpy
            Retorno: 
                - Matriz normalizada conforme SNV
        """


        out = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            out[i, :] = (matrix[i, :] - np.mean(matrix[i, :])) / np.std(matrix[i, :])

        return out

    @staticmethod
    def normalize_mean(matrix: np.ndarray):
         """
            Centralizar os dados em 0 com a média
            
            Parâmetros: 
                - matrix: Matriz com formato numpy
            Retorno: 
                - Matriz normalizada
        """

        out = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            out[i, :] = (matrix[i, :] - np.mean(matrix[i, :]))

        return out

    @staticmethod
    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        
        """
            Savitzky-Golay para atenuação dos ruídos e conservação do sinal de interesse. 
            
            Parâmetros: 
                - y: O dado filtrado (saida)
                - window_size: O comprimento da janela do filtro (ou seja, o número de coeficientes). 
                  Se o modo for ‘interp’, window_length deve ser menor ou igual ao tamanho de x.
                - order: A ordem do polinômio usado para ajustar as amostras. Order deve ser menor que window_size.
                - deriv: A ordem da derivada a ser calculada. Este deve ser um número inteiro não 
                  negativo. O padrão é 0, o que significa filtrar os dados sem diferenciar.
                - mode='constant': A extensão contém o valor fornecido pelo argumento cval (Valor a ser preenchido além 
                  das bordas da entrada se o modo for ‘constante’. O padrão é 0,0.).
            Retorno: 
                - A matrix y filtrada 
        """
        order_range = range(order + 1)
        half_window = (window_size - 1) // 2

        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)

        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])

        y = np.concatenate((firstvals, y, lastvals))

        return np.convolve(m[::-1], y, mode='valid')

    @staticmethod
    def sgolay(matrix: np.ndarray, order=2, window=21, derivative=1, mode='wrap'):
        """
            Savitzky-Golay filter
        """
        return savgol_filter(matrix, window, order, deriv=derivative, mode=mode)

    def removeBg(self, matrix, pcs):
        """
            Matriz 3d para remover o bg
            pcs: número de pcs para kmeans
            
            Parâmetros: 
                - matrix: Matriz de entrada
                - pcs: Número de componentes principais
            Retorno: 
                - Matriz filtrada com a remoção do background
        """
        scores = PCA(n_components=pcs).fit_transform(self.hsi2matrix(matrix))
        return KMeans(n_clusters=2, init='k-means++', n_init=5, max_iter=300).fit(scores).labels_

    @staticmethod
    def rgbscale(image):
        
         """
            Converter imagem para escala em RGB    
            
            Parâmetros: 
                - image: Imagem gerada na função acima (removeBg)
            Retorno:
                - Imagem convertida em RGB
        """
        return (image * 255).astype(np.uint8)

    @staticmethod
    def realIdx(idx, c):
        out = np.arange(idx.shape[0])
        for idx, (rid, vec) in enumerate(zip(out, idx)):
            if vec != c:
                out[idx] = -1
        out[out == -1] = 0

        return out

    @staticmethod
    def sum_idx_array(idx):
        ind_r = []
        for i, (j, ind) in enumerate(zip(idx, np.arange(idx.shape[0]))):
            if j != ind:
                ind_r.append(i)

        return np.delete(idx, ind_r), np.array(ind_r)

    @staticmethod
    def rev_idx_array(idx, rmv, shape=None, tfill=None):
        """
           Criar um array de idx de acordo com
           idx e rmv, matrizes de índices
        """

        if shape is None:
            out = np.zeros(idx.shape[0] + rmv.shape[0])
        else:
            out = np.zeros(shape)

        out[rmv] = 0

        if tfill is not None:
            for i, row in enumerate(idx):
                out[row] = tfill[i]
        else:
            out[idx] = 1

        return out.astype(int)

    def getCluster(self, image, idx, c, rgb):
          """  
            Apresentar o idx na imagem
        """

        ind = self.realIdx(idx, c)
        out_i = np.concatenate((ind, ind, ind), axis=0).reshape((3, *(image.shape[:2])))

        if len(image.shape) == 2:
            image = MinMaxScaler(feature_range=(0, 1)).fit_transform(image)
            image = np.stack((image, image, image), axis=2)

        image[out_i[0] != 0, 0] = rgb[0]
        image[out_i[1] != 0, 1] = rgb[1]
        image[out_i[2] != 0, 2] = rgb[2]

        return image


if __name__ == '__main__':
    routine = HsiRoutine()
    print('so far so good')
