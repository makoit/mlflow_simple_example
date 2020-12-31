# mlflow_simple_example

**project shows usage of mlflow (simple case)**

## run project

create virtual python env

```shell
python -m venv .env
```

activate env

```shell
source .env/Scripts/activate
```

install deps

```shell
pip install -r requirements.txt
```

---

## run the training

```shell
python train_elasticnet.py
```

---

## use local log path (without tracking server)

- after run local `.\mlruns ` folder will be created (includes the logging and the artifacts)
- run in command line: `mlflow ui` and open `localhost:5000' in a browser to see the mlflow results

---

## use mlflow servtracking server ([docs](https://mlflow.org/docs/latest/tracking.html))

- mlflow tracking server has two components for storage: a backend store and an artifact store

**backend store:**

- the backend store is where mlflow tracking server stores experiment and run metadata as well as params, metrics, and tags for runs
- mlflow supports two types of backend stores: file store and database-backed store
  - file store backend as `./path_to_store` or `file:/path_to_store`
  - database-backed store as SQLAlchemy database URI. The database URI typically takes the format `<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>` (mlflow supports the database dialects mysql, mssql, sqlite, and postgresql)
- use `--backend-store-uri` to configure the type
- by default --backend-store-uri is set to the local ./mlruns directory (the same as when running mlflow run locally)

**artifact store:**

- supported stores:

  - local file paths
  - Amazon S3 and S3-compatible storage
  - Azure Blob Storage
  - Google Cloud Storage
  - FTP server
  - SFTP Server
  - NFS
  - HDFS

- is the location for large data and is where clients log their artifact output
- use `--default-artifact-root` (defaults to local `./mlruns` directory) to configure default location to server’s artifact store
- client directly pushes artifacts to the artifact store, it does not proxy these through the tracking server

**base example** for set up a tracking server with sqlite db and local path for artifacts:

```shell
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./artifacts \
--host 0.0.0.0
```

- after running the command `artifacts` folder (include models) and local `mlflow.db` file (include logs) based db will be created

- results of the run can be found on `localhost:5000' -> mlflow web ui

**to use the tracking server -> add code to train script:**

```python
tracking_uri = 'http://localhost:5000'
mlflow.set_tracking_uri(tracking_uri)
```

---

## serving a model via REST interface

- deploy a local REST server that can serve predictions (change the run id for your serving)

```shell
mlflow models serve -m ./artifacts/0/<run_id>/artifacts/model  -p 1234
```

- call api for inference

```shell
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data "{\"columns\":[\"alcohol\", \"chlorides\", \"citric acid\", \"density\", \"fixed acidity\", \"free sulfur dioxide\", \"pH\", \"residual sugar\", \"sulphates\", \"total sulfur dioxide\", \"volatile acidity\"],\"data\":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}" http://127.0.0.1:1234/invocations
```

- sends back prediction in this schema:

```shell
[4.479576972663602]
```

---

## use mlflow projects ([docs](https://mlflow.org/docs/latest/projects.html))

- mlflow project is a format for packaging data science code in a reusable and reproducible way
- adding a MLproject file (YAML formatted textfile which describes the project in more detail):
  - **name**: project name
  - **entry points**: commands that can be run and information about the parameters e.g. `train.py`
  - **env**: defines the software environment which is required for running the project (conda or docker)
- adding env files:

  - **supports:** `conda`, `docker` and `system` envs
  - **conda**: support both python packages and native libraries, mlflow uses the system path to find and run the conda binary
  - **docker**:
    - when you run an MLflow project that specifies a Docker image, MLflow adds a new Docker layer that copies the project’s contents into the `/mlflow/projects/code` directory -> this step produces a new image -> mlflow then runs the new image and invokes the project entrypoint in the resulting container
    - environment variables, such as `MLFLOW_TRACKING_URI`, are propagated inside the Docker container during project execution

- run the project with mlflow cli (inside the project folder)
- creates local logging folder `./mlruns` (do not use `tracking_uri` in the script)

e.g.

```shell
mlflow run . -P alpha=0.5
```

e.g. run with current local system env

```shell
mlflow run . -P alpha=0.5 --no-conda
```

---

## mlflow models ([docs](https://mlflow.org/docs/latest/models.html))

- mlflow model is a standard format for packaging ml models that can be used in a variety of downstream tools e.g. real-time serving via rest api or batch inference via apache spark)
- model defines a convention to save a model in different flavors -> can be understood by different tools

**storage format**:

- models are saved in dirs containing some files
- `MLmodel` file in the root dir describes the model and it's different flavors
- flavors are the key concept
- mlflow defines several standard flavors (e.g. `python function` flavor -> describes how to run the model as a python function OR `mlflow.sklearn` allows loading models back as scikit-learn Pipline)

Directory written by `mlflow.sklearn.save_model(model, "my_model")`:

```
my_model/
├── MLmodel
└── model.pkl
```

... and its Mlmodel file describes the two flavors:

```
time_created: 2018-05-25T17:28:53.35

flavors:
  sklearn:
    sklearn_version: 0.19.1
    pickled_model: model.pkl
  python_function:
    loader_module: mlflow.sklearn
```

**model api:**

- `add_flavor`: to add a flavor to the model. Each flavor has a string name and a dictionary of key-value attributes, where the values can be any object that can be serialized to YAML.

- `save`: to save the model to a local directory.

- `log`: to log the model as an artifact in the current run using MLflow Tracking.

- `load`: to load a model from a local directory or from an artifact in a previous run.

**built-in model flavors:**

- Python Function (python_function)

- R Function (crate)

- H2O (h2o)

- Keras (keras)

- MLeap (mleap)

- PyTorch (pytorch)

- Scikit-learn (sklearn)

- Spark MLlib (spark)

- TensorFlow (tensorflow)

- ONNX (onnx)

- MXNet Gluon (gluon)

- XGBoost (xgboost)

- LightGBM (lightgbm)

- Spacy(spaCy)

- Fastai(fastai)

- Statsmodels (statsmodels)

**built-in deployment tools:**

- deploy MLflow models locally as local REST API endpoints or to directly score files or mlflow can package models as self-contained Docker images with the REST API endpoint

- deploy a python_function model on Microsoft Azure ML

- eploy a python_function model on Amazon SageMaker

- export a python_function model as an Apache Spark UDF

---

## mlflow model registry

- component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model

- It provides model lineage (which MLflow experiment and run produced the model), model versioning, stage transitions (for example from staging to production), and annotations

## mlflow plugins ([docs](https://mlflow.org/docs/latest/plugins.html))

- **Tracking Store**: override tracking backend logic, e.g. to log to a third-party storage solution

- **ArtifactRepository**: override artifact logging logic, e.g. to log to a third-party storage solution

- **Run context providers**: specify context tags to be set on runs created via the mlflow.start_run() fluent API.

- **Model Registry Store**: override model registry backend logic, e.g. to log to a third-party storage solution

- **MLFlow Project backend**: override the local execution backend to execute a project on your own cluster (Databricks, kubernetes, etc.)

#### Community Plugins

- e.g. SQL Server Plugin: allows MLflow to use a relational database as an artifact store
- ...
