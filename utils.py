import os
import pickle
import numpy as np


class Utils:
    def __init__(self):
        self.colors = {
            '0': '#2001c4',
            '1': '#707e51',
            '2': '#df0100',
            '3': '#a96a77',
            '4': '#288bb0',
            '5': '#643a89',
            '6': '#715657',
            '7': '#533e55',
            '8': '#68be9e',
            '9': '#bafeac',
            '10': '#bf4f37',
            '11': '#0fd1c6',
            '12': '#22cbe3',
            '13': '#961e53',
            '14': '#1ad397',
            '15': '#811771',
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
        return self.__colors

    @colors.setter
    def colors(self, var):
        self.__colors = var

    @staticmethod
    def load_samples(folder):
        return [a for a in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, a))]

    @staticmethod
    def get_name(samples_dict, group, case, sz=2):
        labels = [key for key in samples_dict.keys() if samples_dict[key][case] == group]

        return ''.join(labels[0].split('_')[:sz])

    @staticmethod
    def load_bacteria(path: str, name: str, folder='capture'):
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
        samples_dict = {}
        for sample, idx in zip(samples, np.arange(len(samples))):
            samples_dict[sample] = [idx + 1]

        return samples_dict

