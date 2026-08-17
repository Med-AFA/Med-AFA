# Med-AFA

## Background

Med-AFA is a research implementation for clinically constrained active feature acquisition in progressive medical prediction. Medical variables are organized into clinically meaningful action groups, and prerequisite relations between actions are enforced during acquisition. The method learns to update diagnostic predictions from partially observed patient information while selecting the next legal action according to short-term and long-term diagnostic value.

## Public Data Sources

The experiments use the following public datasets. Please obtain the data from the original providers and comply with their respective terms of use.

- Heart Disease: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Chronic Kidney Disease Dataset: [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/chronic-kidney-disease-dataset-analysis)
- Cirrhosis Patient Survival Prediction: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1)
- Dermatology: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/33/dermatology)

## Installation

Python 3.11 is recommended. From the project root, create the environment and install the dependencies:

```powershell
conda create -n med_afa python=3.11 -y
conda activate med_afa
python -m pip install --upgrade pip
pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

The CUDA-enabled PyTorch command requires a compatible NVIDIA driver. See [environment_setup.txt](environment_setup.txt) for the same installation steps and GPU availability check.
