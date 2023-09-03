# SpeakerDiarization
이 프로젝트는 화자를 인식하고 구별하는 것을 목표로 합니다.

## 주요 기능
### 화자 인식
MFCC기법을 이용하여 화자 발언의 특징 값을 추출하고 추출된 특징 값으로부터 GMM모델을 생성합니다. 그 후 각 모델별 스코어링을 통해서 화자를 특정 합니다. 
### 화자 구분
특정 프레임크기와 프레임이동 크기를 정해서 추출된 MFCC값을 프레임별로 모델별 스코어링을 하여 저장합니다. 저장된 값을 분석하여 화자의 발언 시작점과 끝점을 찾아 서로 다른 화자를 구분합니다.

## 샘플 데이터 소스
* https://github.com/goodatlas/zeroth
* http://www.openslr.org/40/

## Licence
* License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

# Speaker Diarization
This project aims to recognize and distinguish speakers.

## Key Features
### Speaker Recognition
Using the MFCC technique, it extracts feature values from speaker utterances and creates GMM models from the extracted feature values. Subsequently, it identifies speakers through scoring for each model.

### Speaker Segmentation
By specifying frame sizes and frame shift sizes, it extracts MFCC values and performs frame-wise scoring for each model, storing the results. It then analyzes the stored values to find the starting and ending points of speaker utterances, distinguishing different speakers.