## 실행 환경
```
Ubuntu 22.04.3
cuDNN 9.4.0.58
CUDA 12.6
Python 3.10.12
PyTorch 2.4.1
```
## Requirements
```
tensorflow==2.11.1
keras-cv==0.4.2
albumentations==1.3.0
transformers==4.27.4
scikit-learn==1.2.2
numpy==1.24.2
Cython==0.29.34
pycocotools==2.0.6
pandas==2.0.0
jupyter==1.0.0
notebook==6.5.3
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.14.0
Pillow==9.5.0
opencv-python==4.7.0.72
scikit-image==0.20.0
```

## 실행 방법
- 데이터 전처리 및 증강 & Quantization-Aware Training 실행
    - `sh train.sh`
    - 훈련한 모델은 `{root}/Baseline_Model/model/{version}`에 저장됩니다.


- Quantized Model 테스트 실행
	- `sh test.sh`
        - default: 350 steps checkpoint
    - 다른 체크포인트를 사용하는 방법
        ```
        cd Baseline_Model
        python test.py —checkpoint {path}
        ```
