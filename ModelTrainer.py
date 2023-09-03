import pickle
import numpy as np

from os import listdir
from os.path import isfile, join

from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
from FeatureExtractor import FeatureExtractor
from collections import defaultdict


class ModelTrainer:
    def __init__(self):
        # file_info dictionary {speaker_name : {file_paths:[], cnt:0}}
        self.file_infos = defaultdict(lambda: {"file_paths": [], "cnt": 0})

    def loadFiles(self, dir_path):
        """Read wav files from directory path
        The file format of the folder must be something like "speaker_num.wav".
        It is then saved in the file_infos variable in the following format: "{speaker_name:{file_paths:[],cnt:0}}".
        """
        list_dir = sorted([f for f in listdir(dir_path) if isfile(join(dir_path, f))])
        for f in list_dir:
            speaker_name = f.split("_")[0]
            self.file_infos[speaker_name]["file_paths"].append(join(dir_path, f))
            self.file_infos[speaker_name]["cnt"] += 1

    def train(self, dir_path):
        """Train the model and save results to a directory path

        Args:
            dir_path: This is the path to save the learning results.
        """
        features = np.asarray(())
        FE = FeatureExtractor()
        for speaker_name, value in self.file_infos.items():
            file_paths = value["file_paths"]
            for i, file_path in enumerate(file_paths):
                # read the audio
                sr, audio = read(file_path)

                # extract 40 dimensional MFCC & delta MFCC features
                vector = FE.extract_features(audio, sr)

                if i == 0:
                    features = vector
                else:
                    features = np.vstack((features, vector))
                # Once all features in the speaker file are concatenated, training of the model begins.
                if i == len(file_paths) - 1:
                    gmm = GMM(
                        n_components=20, max_iter=200, covariance_type="diag", n_init=3
                    )
                    gmm.fit(features)

                    # dumping the trained gaussian model
                    picklefile = speaker_name + ".gmm"
                    pickle.dump(gmm, open(dir_path + picklefile, "wb"))
                    print(
                        "+ modeling completed for speaker:",
                        picklefile,
                        " with data point = ",
                        features.shape,
                    )
                    features = np.asarray(())


if __name__ == "__main__":
    mt = ModelTrainer()
    mt.loadFiles("./train_data/")
    mt.train("./test_data/")
