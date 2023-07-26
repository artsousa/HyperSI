import os
import pickle
import spectral as sp

sp.settings.envi_support_nonlowercase_params = True


""" 
    Class Sample responsável por preparar os arquivos obtidos em Utils e gerar os hipercubos necessários. 
    Os arquivos gerados pela câmera (DARK, WHITE e .hdr) seguem a estrutura de arquivos dentro a "capture" e são 
    passados para o formato numpy.
 
 """


import os
import pickle
import spectral as sp

sp.settings.envi_support_nonlowercase_params = True


class Sample:
    def __init__(self, sample_name, folder: str = "capture", sample_prefix: str = None):
        """
        Args:
            sample_name: The path to the sample.
            inter: The interface used to capture the sample.
            sample_prefix: The prefix of the sample.
            to_numpy_format: Whether to convert the sample to NumPy format.
        """

        self.path = os.path.join(sample_name, (folder if folder else ""))
        self.sample_name = (sample_prefix if sample_prefix else "") + sample_name.split(
            os.sep
        )[-1]

        self.image = None
        self.sample = None
        self.processed = None
        self.normalized = None
        self.sample_cluster = None

        self._read_image()

    def _read_image(
        self,
    ):
        """
            Reads the image and stores it in the `sample` attribute.
        """
        try:
            self.image = sp.open_image(
                os.path.join(self.path, self.sample_name + ".hdr")
            )

            self.sample = self.image.load().transpose(2, 0, 1)

        except Exception as e:
            print(e)

    def save(self):
        """
            Saves the sample to a pickle file.
        """
        sample_path = os.path.join(self.path, self.sample_name)
        sample_file = sample_path + ".pkl"

        with open(sample_file, "wb") as destination_dir:
            pickle.dump(self, destination_dir, -1)

    @property
    def image(self):
        """
            Returns the image.
        """
        return self.__image

    @image.setter
    def image(self, var):
        """
            Sets the image.
        """
        self.__image = var

    @property
    def sample(self):
        """
            Returns the sample.
        """
        return self.__sample

    @sample.setter
    def sample(self, var):
        """
            Sets the sample.
        """
        self.__sample = var

    @property
    def normalized(self):
        """
            Returns the normalized sample.
        """
        return self.__normalized

    @normalized.setter
    def normalized(self, var):
        """
            Sets the normalized sample.
        """
        self.__normalized = var

    @property
    def processed(self):
        """
            Returns the processed sample.
        """
        return self.__processed

    @processed.setter
    def processed(self, var):
        """
            Sets the processed sample.
        """
        self.__processed = var

    @property
    def sample_cluster(self):
        """
            Returns the sample cluster.
        """
        return self.__sample_cluster

    @sample_cluster.setter
    def sample_cluster(self, var):
        """
            Sets the sample cluster.
        """
        self.__sample_cluster = var


if __name__ == '__main__':
    sample = Sample('dir',
                    'Enterobacteaerogenes_13048_Plastico_A_Contaminado_180926-102646')

    print(sample.image.shape)
    print('done')
