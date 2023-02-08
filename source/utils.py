import os
import pickle
import numpy as np


class Utils:
    def __init__(self):
        self.colors = {
            '0': '#73b7df',
            '1': '#ff0000',
            '2': '#2001c4',
            '3': '#a96a77',
            '4': '#2a5ba7',
            '5': '#c8f9a1',
            '6': '#eec78b',
            '7': '#32e726',
            '8': '#87508b',
            '9': '#253b85',
            '10': '#ff6a00',
            '11': '#6dca0d',
            '12': '#1b8991',
            '13': '#a16180',
            '14': '#1ad397',
            '15': '#2001c4',
            '16': '#404686',
            '17': '#4c42e2',
            '18': '#fbf899',
            '19': '#bdd387',
            '20': '#7774bd',
            '21': '#1b0f3f',
            '22': '#32e726',
            '23': '#b25e1c',
            '24': '#87508b',
            '25': '#fa38ff',
            '26': '#c0e33c',
            '27': '#6b8a93',
            '28': '#cec15f',
            '29': '#7cbca0',
            '30': '#692225',
            '31': '#4e7aee',
            '32': '#89f41d',
            '33': '#2a5ba7',
            '34': '#5cb70b',
            '35': '#c8f9a1',
            '36': '#cbc184',
            '37': '#253b85',
            '38': '#919b65',
            '39': '#76929b',
            '40': '#7e6943',
            '41': '#7b1170',
            '42': '#2785ca',
            '43': '#a16180',
            '44': '#45abc2',
            '45': '#eec78b',
            '46': '#f8310a',
            '47': '#1b8991',
            '48': '#a5c7a5',
            '49': '#d67f3a',
            '50': '#6dca0d',
            '51': '#139386',
            '52': '#3cf0ff',
            '53': '#4f8013',
            '54': '#4c1134',
            '55': '#c28f0d',
            '56': '#2ddfdb',
            '57': '#eaadda',
            '58': '#dcd64b',
            '59': '#c0a95c',
            '60': '#b375f6',
            '61': '#73b7df',
            '62': '#2a5b84',
            '63': '#d34e2d',
        }

    @property
    def colors(self):
        """
            Função colors com objetivo de retornar as cores previamente declaradas
        """
        return self.__colors

    @colors.setter
    def colors(self, var):
        """
            Função colors com objetivo de setar as cores previamente declaradas
        """
        self.__colors = var

    @staticmethod
    def load_samples(folder):
        """ 
            Parâmetros: 
                - Folder: Caminho diretório em que se encontra os arquivos
            Retorno:
                - O nome de todos dos diretórios contidos. 
        """
        return [a for a in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, a))]

    @staticmethod
    def get_name(samples_dict, group, case, sz=2):
        
        """
           Função "get_name" com objetivo de retornar o nome de cada amostra, tratado de maneira a obter o nome da amostra 
           sem "_" e com os primeiros digitos para identificação.
           
            Parâmetros: 
                - samples_dict: Dicionário com rótulos das amostras de bactérias
                - group: Especificação da amostra sendo entre especie ou coloração
                - case: Caso de trabalho, de acordo com o dicionário de amostras (case=0 - espécies /case=1 - coloração)
            Retorno:
                - Label com o nome da amostra tratado. Ex. Enterobacteaerogenes13048
        """
        
        labels = [key for key in samples_dict.keys() if samples_dict[key][case] == group]

        return ''.join(labels[0].split('_')[:sz])

    @staticmethod
    def load_hsisample(path: str, name: str, folder='capture'):
        
        """
        Função "load_hsisample" com objetivo de obter o caminho das amostras e retornar o carregamento do arquivo.

        Parâmetros: 
            - path: Diretório 
            - name: Nome da amostra (conforme o nome que consta no diretório)
            - folder: Todos arquivos encontram-se dentro do diretório capture 
            (Ex. Plastico B\Klebsielapneumonial_700603_Plastico_B_180926-105913\capture)
        Retorno: 
            - Saída do diretório
        """
        sample_path = os.path.join(path, name)
        with open(os.path.join(sample_path, folder, name + '.pkl'), 'rb') as file:
            out = pickle.load(file)

        return out

    @staticmethod
    def hex2rgb(value):
        return tuple(int(value.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def no_rep(names: list):
        return np.array(list(set(names)))

    @staticmethod
    def get_dict(samples: list):
        """
        Função "get_dict" tem como objetivo retornar o dicionário com o nome das amostras.
        
            Parâmetros: 
                - Lista com as amostras contidas.
        """
        samples_dict = {}
        for sample, idx in zip(samples, np.arange(len(samples))):
            samples_dict[sample] = [idx + 1]

        return samples_dict

