# MLFLOW Model Deployment in CML with APIv2

Cloudera Machine Learning (CML) on Cloudera Data Platform (CDP) accelerates time-to-value by enabling data scientists to collaborate in a single unified platform that is all inclusive for powering any AI use case. Purpose-built for agile experimentation and production ML workflows, Cloudera Machine Learning manages everything from data preparation to MLOps, to predictive reporting.

CML exposes a REST API that you can use to perform operations related to projects, jobs, and runs. You can use API commands to integrate CML with third-party workflow tools or to control CML from the command line.

In this example we will showcase how to use APIv2 to programmatically register an XGBoost experiment via MLFlow Tracking, Registry, and deploy it as a model endpoint in CML.

## MLOps in CML

CML has extensive MLOps features and a set of model and lifecycle management capabilities to enable the repeatable, transparent, and governed approaches necessary for scaling model deployments and ML use cases. It´s built to support open source standards and is fully integrated with CDP, enabling customers to integrate into existing and future tooling while not being locked into a single vendor.

CML enables enterprises to proactively monitor technical metrics such as service level agreements (SLA) adherence, uptime, and resource use as well as prediction metrics including model distribution, drift, and skew from a single governance interface. Users can set custom alerts and eliminate the model “black box” effect with native tools for visualizing model lineage, performance, and trends. Some of the benefits with CML include:

* Model cataloging and lineage capabilities to allow visibility into the entire ML lifecycle, which eliminates silos and blind spots for full lifecycle transparency, explainability, and accountability.
* Full end-to-end machine learning lifecycle management that includes everything required to securely deploy machine learning models to production, ensure accuracy, and scale use cases.
* An extensive model monitoring service designed to track and monitor both technical aspects and accuracy of predictions in a repeatable, secure, and scalable way.
* New MLOps features for monitoring the functional and business performance of machine learning models such as detecting model performance and drift over time with native storage and access to custom and arbitrary model metrics; measuring and tracking individual prediction accuracy, ensuring models are compliant and performing optimally.
* The ability to track, manage, and understand large numbers of ML models deployed across the enterprise with model cataloging, full lifecycle lineage, and custom metadata in Apache Atlas.
* The ability to view the lineage of data tied to the models built and deployed in a single system to help manage and govern the ML lifecycle.
Increased model security for Model REST endpoints, which allows models to be served in a CML production environment without compromising security.

## Use Case

In this example we will create a basic MLOps pipeline to put a credit card fraud classifier into production. We will create a model prototype with XGBoost, register and manage experiments with MLFlow Tracking, and stage the best experiment run in the MLFlow Registry.

Next, we will deploy the model from the Registry into an API Endpoint, and finally redeploy it with additional resources for High Availability and increased serving performance.

The full code is available in this [git repository](https://github.com/pdefusco/cml_api_articles/tree/main).

## Step by Step Guide

#### Setup

Create a CML Project with Python 3.9 / Workbench Editor Runtime.
Launch a CML Session and install requirements.
Open script "00_datagen.py" and update lines 140 and 141 with your Iceberg database name and Spark Data Connection Name. Then run it.

#### Script 1: Create the Model Experiment

Run script "01_train_xgboost.py" in order to create an MLFlow Experiment.

Code highlights:
* MLFlow is installed in CML by default. You must import mlflow in order to use it in your script.
* The experiment run is determined by the "mlflow tracking run context".
* When this executes for the first time an experiment is created. If the same code runs again without changing EXPERIMENT_NAME, a new Experiment Run is logged for the same experiment. Else, a new experiment is created.
* You can log one or multiple metrics for the specific run with the "mlflow.log_param" method.
* Model artifacts such as useful metadata and dependencies are logged with the "mlflow.log_model" method.

```
import mlflow

EXPERIMENT_NAME = "xgb-cc-fraud-{0}-{1}".format(USERNAME, DATE)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():

  model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

  model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
  y_pred = model.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)

  mlflow.log_param("accuracy", accuracy)

  mlflow.xgboost.log_model(model, artifact_path="artifacts")
```

* You can use the mlflow library or instantiate an mlflow client to manage experiments.
* In this example we use the "mlflow.get_experiment_by_name()", "mlflow.search_runs()" and "mlflow.get_run()" methods.
* In this example we also instantiate the client to list artifacts for a specific run.

```
def getLatestExperimentInfo(experimentName):
    """
    Method to capture the latest Experiment Id and Run ID for the provided experimentName
    """
    experimentId = mlflow.get_experiment_by_name(experimentName).experiment_id
    runsDf = mlflow.search_runs(experimentId, run_view_type=1)
    experimentId = runsDf.iloc[-1]['experiment_id']
    experimentRunId = runsDf.iloc[-1]['run_id']

    return experimentId, experimentRunId

experimentId, experimentRunId = getLatestExperimentInfo(EXPERIMENT_NAME)

#Replace Experiment Run ID here:
run = mlflow.get_run(experimentRunId)

pd.DataFrame(data=[run.data.params], index=["Value"]).T
pd.DataFrame(data=[run.data.metrics], index=["Value"]).T

client = mlflow.tracking.MlflowClient()
client.list_artifacts(run_id=run.info.run_id)
```

#### Script 2: Register the Model Experiment

Run script "02_cml_api_endpoint.py" in order to register an MLFlow Experiment.

Code highlights:
* CML APIv2 is installed in your workspace by default. You must import cmlapi in order to use it in your script.
* The API provides about 100 python methods for MLOps.
* In this example, we created a "registerModelFromExperimentRun" method as a wrapper to the API's create_registered_model() method.
* In this example, we created a ModelRegistration class including the "registerModelFromExperimentRun" method to register the model.
* Creating your own Python classes and methods to implement the API methods in the context of your project is highly recommended.

```
import cmlapi
from cmlapi.rest import ApiException

class ModelRegistration():
    """
    Class to manage the model deployment of the xgboost model
    """

    def __init__(self, username, experimentName):
        self.client = cmlapi.default_client()
        self.username = username
        self.experimentName = experimentName

    def registerModelFromExperimentRun(self, modelName, experimentId, experimentRunId, modelPath):
        """
        Method to register a model from an Experiment Run
        Input: requires an experiment run
        Output: api_response object
        """

        model_name = 'xgb-cc-' + username

        CreateRegisteredModelRequest = {
                                        "project_id": os.environ['CDSW_PROJECT_ID'],
                                        "experiment_id" : experimentId,
                                        "run_id": experimentRunId,
                                        "model_name": modelName,
                                        "model_path": modelPath
                                       }

        try:
            # Register a model.
            api_response = self.client.create_registered_model(CreateRegisteredModelRequest)
            #pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_registered_model: %s\n" % e)

        return api_response
```

* The response from the method request contains very useful information.
* In this example, the registeredModelResponse response includes modelId and modelVersionId variables which are in turn used by other API methods.

```
modelReg = ModelRegistration(username, experimentName)

modelPath = "artifacts"
modelName = "FraudCLF-" + username

registeredModelResponse = modelReg.registerModelFromExperimentRun(modelName, experimentId, experimentRunId, modelPath)

modelId = registeredModelResponse.model_id
modelVersionId = registeredModelResponse.model_versions[0].model_version_id
registeredModelResponse.model_versions[0].model_version_id
```

#### Script 3: Deploy Endpoint from Registry Model

Run script "03_api_deployment.py" in order to create a Model Endpoint from the registered model.

Code highlights:
* In this example we created a ModelDeployment class to manage multiple API wrapper methods.
* The "listRegisteredModels()" method is a wrapper to the API's "list_registered_models()" method. Notice it is arbitrarily preconfigured to list models corresponding to your user and model name. In the context of a broader MLOps pipeline, these values can obviously be parameterized. This method is necessary for obtaining the "registeredModelId" variable needed for model deployment.
* The "getRegisteredModel()" method is a wrapper to the API's "get_registered_model()" method. This method is necessary for obtaining the "modelVersionId" variable needed for model deployment.
* Once registeredModelId and modelVersionId are obtained, you can begin the deployment. The deployment consists of three phases: model creation, model build, and model deployment.
* The model creation corresponds to the creation of an API Endpoint. Once you run this, you will see a new entry in the Model Deployments tab.
* The model build corresponds to the creation of the model's container. Thanks to MLFlow Registry, CML automatically packages all dependencies used to train the Experiment into the model endpoint for you.
* The model deployment corresponds to the activation of the model endpoint. This is when the container with its associated resource profile and endpoint is actually deployed so inference can start.
* The "listRuntimes()" method is an example of querying the Workspace for all available runtimes in order to select the most appropriate for model build.

```
class ModelDeployment():
    """
    Class to manage the model deployment of the xgboost model
    """

    def __init__(self, projectId, username):
        self.client = cmlapi.default_client()
        self.projectId = projectId
        self.username = username

    def listRegisteredModels(self):
        """
        Method to retrieve registered models by the user
        """

        #str | Search filter is an optional HTTP parameter to filter results by. Supported search_filter = {\"model_name\": \"model_name\"}  search_filter = {\"creator_id\": \"<sso name or user name>\"}. (optional)

        search_filter = {"creator_id" : self.username, "model_name": "FraudCLF-"+self.username}
        search = json.dumps(search_filter)
        page_size = 1000

        try:
            # List registered models.
            api_response = self.client.list_registered_models(search_filter=search, page_size=page_size)
            #pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->list_registered_models: %s\n" % e)

        return api_response


    def getRegisteredModel(self, modelId):
        """
        Method to return registered model metadata including model version id
        """
        search_filter = {"creator_id" : self.username}
        search = json.dumps(search_filter)

        try:
            # Get a registered model.
            api_response = self.client.get_registered_model(modelId, search_filter=search, page_size=1000)
            #pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->get_registered_model: %s\n" % e)

        return api_response


    def createModel(self, projectId, modelName, registeredModelId, description = "Fraud Detection 2024"):
        """
        Method to create a model
        """

        CreateModelRequest = {
                                "project_id": projectId,
                                "name" : modelName,
                                "description": description,
                                "registered_model_id": registeredModelId,
                                "disable_authentication": True
                             }

        try:
            # Create a model.
            api_response = self.client.create_model(CreateModelRequest, projectId)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model: %s\n" % e)

        return api_response


    def createModelBuild(self, projectId, modelVersionId, modelCreationId, runtimeId):
        """
        Method to create a Model build
        """


        # Create Model Build
        CreateModelBuildRequest = {
                                    "registered_model_version_id": modelVersionId,
                                    "runtime_identifier": runtimeId,
                                    "comment": "invoking model build",
                                    "model_id": modelCreationId
                                  }

        try:
            # Create a model build.
            api_response = self.client.create_model_build(CreateModelBuildRequest, projectId, modelCreationId)
            #pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model_build: %s\n" % e)

        return api_response


    def createModelDeployment(self, modelBuildId, projectId, modelCreationId):
        """
        Method to deploy a model build
        """

        CreateModelDeploymentRequest = {
          "cpu" : "2",
          "memory" : "4"
        }

        try:
            # Create a model deployment.
            api_response = self.client.create_model_deployment(CreateModelDeploymentRequest, projectId, modelCreationId, modelBuildId)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model_deployment: %s\n" % e)

        return api_response


    def listRuntimes(self):
        """
        Method to list available runtimes
        """
        search_filter = {"kernel": "Python 3.9", "edition": "Standard", "editor": "Workbench"}
        # str | Search filter is an optional HTTP parameter to filter results by.
        # Supported search filter keys are: [\"image_identifier\", \"editor\", \"kernel\", \"edition\", \"description\", \"full_version\"].
        # For example:   search_filter = {\"kernel\":\"Python 3.7\",\"editor\":\"JupyterLab\"},. (optional)
        search = json.dumps(search_filter)
        try:
            # List the available runtimes, optionally filtered, sorted, and paginated.
            api_response = self.client.list_runtimes(search_filter=search, page_size=1000)
            #pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->list_runtimes: %s\n" % e)

        return api_response
```

Once you have created your model endpoint, give it a minute and then try a test request:

```
{"dataframe_split": {"columns": ["age", "credit_card_balance", "bank_account_balance", "mortgage_balance", "sec_bank_account_balance", "savings_account_balance", "sec_savings_account_balance", "total_est_nworth", "primary_loan_balance", "secondary_loan_balance", "uni_loan_balance", "longitude", "latitude", "transaction_amount"], "data":[[35.5, 20000.5, 3900.5, 14000.5, 2944.5, 3400.5, 12000.5, 29000.5, 1300.5, 15000.5, 10000.5, 2000.5, 90.5, 120.5]]}}
```

#### Script 4: Endpoint Redeployment

Run script "04_api_redeployment.py" in order to create a new model deployment with increased resources.

Code highlights:
* A slightly different version of the ModelDeployment class is implemented. This includes the "get_latest_deployment_details()" as an example of creating a wrapper method to the API's "list_models()" and "list_model_deployments()" methods all in one. You can implement your own methods in a similar fashion as best needed in the context of your MLOps pipeline.
* Once the latest model deployment's metadata has been obtained in one go, a new model build is created with additional CPU, Memory and Replicas. Notice that in the process you also have the ability to switch to a different runtime as needed.

```
deployment = ModelDeployment(projectId, username)

getLatestDeploymentResponse = deployment.get_latest_deployment_details(modelName)

listRuntimesResponse = deployment.listRuntimes()
listRuntimesResponse

runtimeId = 'docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-workbench-python3.9-standard:2024.02.1-b4' # Copy a runtime ID from previous output

cpu = 2
mem = 4
replicas = 2

createModelBuildResponse = deployment.createModelBuild(projectId, modelVersionId, modelCreationId, runtimeId, cpu, mem, replicas)
modelBuildId = createModelBuildResponse.id

deployment.createModelDeployment(modelBuildId, projectId, modelCreationId)
```

## Summary and Next Steps

Cloudera Machine Learning exposes a REST API that you can use to perform operations related to projects, jobs, and runs. You can use API commands to integrate CML with third-party workflow tools or to control CML from the command line.

CML's API accelerates your data science projects by allowing you to build end to end pipelines programmatically. When coupled with CML MLFlow Tracking and MLFlow Registry, it can be used to manage models from inception to production.

* [APIv2 Documentation](https://docs.cloudera.com/machine-learning/cloud/api/topics/ml-api-v2.html)
* [APIv2 Examples](https://docs.cloudera.com/machine-learning/cloud/api/topics/ml-apiv2-usage-examples.html)
* [APIv2 AMP](https://github.com/cloudera/CML_AMP_APIv2)
* [Registering and Deploying Models with Model Registry](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-registering-and-deploying-a-model-registry.html)
* [Securing Models](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-securing-models.html)
* [CML Projects](https://docs.cloudera.com/machine-learning/cloud/projects/topics/ml-collaborate.html)
* [CML Runtimes](https://docs.cloudera.com/machine-learning/cloud/runtimes/topics/ml-runtimes-overview.html)
* [Introducing MLOps and SDX for Models in CML](https://blog.cloudera.com/introducing-mlops-and-sdx-for-models-in-cloudera-machine-learning/)
* [Model Registry GA in CML](https://blog.cloudera.com/announcing-general-availability-of-model-registry/)
