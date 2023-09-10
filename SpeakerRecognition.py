import pickle
import numpy as np
import speech_recognition

from os import listdir
from os.path import isfile, join, basename, splitext
from pydub import AudioSegment
from scipy.io.wavfile import read
from FeatureExtractor import FeatureExtractor


class SpeakerRecognition:
    def __init__(self):
        self.scripts = []
        self.vector = []
        self.models = []
        self.participants = []
        self.predicts = []
        self.ends = []  # The end of each speaker's speech

        self.result_dir = ""
        self.file_path = ""
        self.file_name = ""
        self.new_path_format = ""

        self.is_fail = False

    def loadParticipants(self, file_path):
        """Load participants info from file_path

        Example: participants.txt
            speaker1\n
            speaker2\n
            speaker3\n

            Each speaker is separated by only one line break.
        """
        with open(file_path, "r") as f:
            for s in f:
                s = s.strip()
                self.participants.append(s)

    def loadGmms(self, model_dir):
        """Loads only the gmm models corresponding to conversation participants in the model_dir path."""
        file_names = [f for f in listdir(model_dir) if isfile(join(model_dir, f))]
        file_names = [f for f in file_names if f.split(".gmm")[0] in self.participants]

        gmm_paths = [
            join(model_dir, fname) for fname in file_names if fname.endswith(".gmm")
        ]

        # Load the Gaussian gender Models
        self.models = [pickle.load(open(gmm_path, "rb")) for gmm_path in gmm_paths]

    def recognition(self, participants_path, model_dir, file_path, result_dir):
        """Recognizes voice files in the file path and separates speakers."""
        self.file_path = file_path
        self.file_name = splitext(basename(file_path))[0]
        self.result_dir = result_dir

        self.loadParticipants(participants_path)
        self.loadGmms(model_dir)

        self.is_fail = False
        print("Testing Audio : ", file_path)
        FE = FeatureExtractor()
        sr, audio = read(file_path)
        self.vector = FE.extract_features(audio, sr)

        self.diarization()
        self.wavSeperation()
        self.speechToText()
        self.saveResult()

    def diarization(self):
        winLen = 150
        winStep = 10
        startLen = 100  # The first n vectors are ignored.
        lastLen = 100  # The end n vectors are ignored.

        winners = []
        log_likelihood = np.zeros(len(self.models))
        ##### Scoring #####
        if winLen + startLen + lastLen <= self.vector.__len__():
            # long audio file
            for index in range(self.vector.__len__() // winStep):
                if (
                    index * winStep + winLen + startLen + lastLen
                    >= self.vector.__len__()
                ):
                    break
                for i in range(len(self.models)):
                    gmm = self.models[i]  # checking with each model one by one
                    scores = np.array(
                        gmm.score(
                            self.vector[
                                index * winStep
                                + startLen : index * winStep
                                + startLen
                                + winLen
                            ]
                        )
                    )
                    log_likelihood[i] = scores.sum()

                winner = np.argmax(log_likelihood)
                winners.append((startLen + index * winStep, winner))
                print(
                    startLen + index * winStep,
                    "\tdetected as - ",
                    self.participants[winner],
                )
        else:
            # short audio file
            for i in range(len(self.models)):
                gmm = self.models[i]  # checking with each model one by one
                scores = np.array(gmm.score(self.vector))
                log_likelihood[i] = scores.sum()
            winners.append((0, np.argmax(log_likelihood)))
            print("\tdetected as - ", self.participants[winners[0][1]])
        ##### End Scoring #####

        predicts_tmp = []
        ends_tmp = []
        cnt = 0
        boundary = 15

        tmp = winners[0][1]  # first winner
        if winners.__len__() > 1:
            for fr, win in winners:
                if tmp != win and cnt >= boundary:
                    ends_tmp.append(fr + winLen - boundary * 2)
                    predicts_tmp.append(self.participants[tmp])
                if tmp != win:
                    cnt = 0
                    tmp = win
                cnt += 1
            if cnt >= boundary or predicts_tmp.__len__() == 0:
                predicts_tmp.append(self.participants[tmp])
                ends_tmp.append(self.vector.__len__() - 1)

            ends_tmp[-1] = self.vector.__len__() - 1
            prepredict = ""
            preend = 0
            for i, predict in enumerate(predicts_tmp):
                if prepredict == predict:
                    self.ends.remove(preend)
                    self.predicts.remove(prepredict)
                self.ends.append(ends_tmp[i])
                self.predicts.append(predict)
                prepredict = predict
                preend = ends_tmp[i]
        else:
            self.predicts.append(self.participants[winners[0][1]])

    def wavSeperation(self):
        ##### wav seperation #####
        prev = 0
        if self.predicts.__len__() != 1:
            for i, _ in enumerate(self.predicts):
                sound = AudioSegment.from_mp3(self.file_path)

                # len() and slicing are in milliseconds
                position = float(self.ends[i]) / self.vector.__len__()
                position = round(len(sound) * position)
                if prev - 500 > 0:
                    prev -= 500
                sep = sound[prev:position]
                prev = position
                # writing wav files is a one liner
                self.new_path_format = join(
                    self.result_dir, (self.file_name + "_" + "{index}" + ".wav")
                )
                sep.export(self.new_path_format.format(index=i), format="wav")

    def speechToText(self):
        for i, _ in enumerate(self.predicts):
            r = speech_recognition.Recognizer()
            if self.predicts.__len__() != 1:
                with speech_recognition.WavFile(
                    self.new_path_format.format(index=i)
                ) as source:  # use "test.wav" as the audio source
                    audio_stt = r.record(source)  # extract audio data from the file
            else:
                with speech_recognition.WavFile(
                    self.file_path
                ) as source:  # use "test.wav" as the audio source
                    audio_stt = r.record(source)  # extract audio data from the file
            try:
                script = r.recognize_google(audio_stt, language="ko-KR")
                self.scripts.append(script)
                print(
                    "Predict: " + _
                )  # recognize speech using Google Speech Recognition
                print(
                    "Transcription: " + script
                )  # recognize speech using Google Speech Recognition
            except speech_recognition.UnknownValueError:  # speech is unintelligible
                # file_paths.remove(file_paths[n])
                self.scripts.append("NONE")
                self.is_fail = True
            ##### END STT #####

            if self.is_fail == False:
                # self.predicts.append(self.participants[winner])
                pass
            else:
                self.predicts[i] = "NONE"
                self.is_fail = False

    def saveResult(self):
        ##### Save result.txt #####
        data = ""
        with open(
            join(self.result_dir, ("result_" + self.file_name + ".txt")), "w"
        ) as f:
            for i, _ in enumerate(self.predicts):
                data += "%s, %s\n" % (self.predicts[i], self.scripts[i])
            f.write(data)


if __name__ == "__main__":
    SR = SpeakerRecognition()
    SR.recognition(
        participants_path="./participants.txt",
        model_dir="./models",
        file_path="./test_data/109_2_108_1.wav",
        result_dir="./results",
    )
