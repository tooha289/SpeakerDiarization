import sys

from ModelTrainer import ModelTrainer
from SpeakerRecognition import SpeakerRecognition
if __name__ == '__main__':
    take = int(sys.argv[1])
    file_name = sys.argv[2]
    # take = 1
    if take == 0:
        MT = ModelTrainer()
        MT.loadFiles()
        MT.train()
    elif take == 1:
        SR = SpeakerRecognition()
        SR.recognition(file_name)


