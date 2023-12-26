download:
	kaggle competitions download \
	-c nlp-getting-started \
	-p ./data/raw \
	--force

	unzip ./data/raw/nlp-getting-started.zip -d ./data/raw


install:
	pip install --upgrade pip && pip install -r requirements.txt

format:
	find src -name '*.py' -exec black {} +

lint:
	pylint --disable=R,C src/*.py

test:
	python -m pytest -vv --cov=test


run:
	python run.py


# run_pipeline:
# 	export MLFLOW_RUN_ID=`python run.py $(RUN_NAME)`; \
	

MLFLOW_HOST = 127.0.0.1
MLFLOW_PORT = 7000
MLFLOW_LOCAL_FOLDER = ./data/mlflow

server:
	mlflow server \
	--host $(MLFLOW_HOST) \
	--port $(MLFLOW_PORT) \
	--backend-store-uri $(MLFLOW_LOCAL_FOLDER)