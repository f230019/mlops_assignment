PYTHON=python

RAW=data/raw/titanic.csv
PROCESSED=data/processed/cleaned.csv
FEATURES=features/features.csv
MODEL=models/svm_model.pkl
PREDICTIONS=results/predictions.csv
METRICS=results/metrics.txt

all: setup download-data preprocess features train predict evaluate

setup:
	pip install -r requirements.txt

download-data:
	$(PYTHON) src/download_data.py

preprocess: $(RAW)
	$(PYTHON) src/preprocess.py

features: $(PROCESSED)
	$(PYTHON) src/features.py

train: $(FEATURES)
	$(PYTHON) src/train.py

predict: $(MODEL)
	$(PYTHON) src/predict.py

evaluate: $(PREDICTIONS)
	$(PYTHON) src/evaluate.py

clean:
	rm -rf data/processed/*
	rm -rf features/*
	rm -rf models/*
	rm -rf results/*
