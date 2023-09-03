import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc

class FeatureExtractor:
    def __init__(self):
        self.features = []

    def calculate_delta(self,array):
        """Calculate and returns the delta of given feature vector matrix"""
        rows, cols = array.shape
        deltas = np.zeros((rows, 20))
        N = 2
        for i in range(rows):
            index = []
            j = 1
            while j <= N:
                if i - j < 0:
                    first = 0
                else:
                    first = i - j
                if i + j > rows - 1:
                    second = rows - 1
                else:
                    second = i + j
                index.append((second, first))
                j += 1
            deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
        return deltas

    def extract_features(self, signal, samplerate, winlen=0.025, winstep=0.01, numcep=20):
        """extract 20 dim mfcc features from an audio signal,
          performs CMS and combines delta to make it 40 dim feature vector
        
        Args:
            signal: the audio signal from which to compute features. Should be an N*1 array
            samplerate: the samplerate of the signal we are working with.
            winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
            winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
            numcep: the number of cepstrum to return, default 20
        """
        mfcc_feature = mfcc.mfcc(signal, samplerate, winlen, winstep, numcep, nfft=1200, appendEnergy=True)
        mfcc_feature = preprocessing.scale(mfcc_feature)
        delta = self.calculate_delta(mfcc_feature)
        combined = np.hstack((mfcc_feature, delta))
        self.features = combined
        return combined

    def getMFCC(self):
        return self.features
