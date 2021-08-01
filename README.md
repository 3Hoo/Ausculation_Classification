# Ausculation_Classification

### 사용법
> python3 predict.py --wav_file [file_path] --sample_rate [sample_rate] --model_path ./model/ckpt_best_200_-1.pkl
> 
> ex) python3 predict.py --wav_file ./test/normal001.wav --sample_rate 6000 --model_path ./model/ckpt_best_200_-1.pkl


## 폴더/파일 설명
> ### model
> > 훈련된 RespireNet 모델이 저장되어 있는 폴더입니다

> ### nets
> > RespireNet에 사용된 신경망 구조가 저장되어 있는 폴더입니다

> ### Dockerfile
> > predict.py를 실행시키기 위한 도커 환경을 빌드하는 파일입니다 (opencv-python은 pip으로 추가 설치 해야합니다)

> ### docker_run.txt
> > Docker의 빌드, 실행에 대한 설명이 적혀있는 텍스트 파일입니다

> ### image_dataloader.py, utils.py
> > 각각 데이터 전처리, 각종 유틸 함수가 작성되어 있는 파이썬 파일입니다

> ### predict.py
> > 실행되는 메인 파이썬 파일입니다. 사용법은 위를 참고하시면 됩니다
