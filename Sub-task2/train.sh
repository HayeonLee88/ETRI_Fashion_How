# requirements 설치
pip install -r requirements.txt

printf "\e[34mFin install requirements\e[0m\n"

# Dataset 전처리 및 증강
cd Baseline_Model

python EDA.py
printf "\e[34mFin Preprocessing Dataset\e[0m\n"

# Quantization-Aware Traning
python QAT.py
printf "\e[34mFin Quantization-Aware Training\e[0m\n"
