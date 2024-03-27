#****************************************************************************
# (C) Cloudera, Inc. 2020-2024
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

from __future__ import print_function
import cmlapi
from cmlapi.rest import ApiException
from pprint import pprint
import json, secrets, os, time
import mlflow


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


projectId = os.environ['CDSW_PROJECT_ID']
username = os.environ["PROJECT_OWNER"]
sessionId = secrets.token_hex(nbytes=4)
modelName = "FraudCLF-" + username + "-fromAPI"

deployment = ModelDeployment(projectId, username)

listRegisteredModelsResponse = deployment.listRegisteredModels()
registeredModelId = listRegisteredModelsResponse.models[0].model_id

getRegisteredModelResponse = deployment.getRegisteredModel(registeredModelId)
modelVersionId = getRegisteredModelResponse.model_versions[0].model_version_id

createModelResponse = deployment.createModel(projectId, modelName, registeredModelId)
modelCreationId = createModelResponse.id

listRuntimesResponse = deployment.listRuntimes()
listRuntimesResponse

runtimeId = 'docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-workbench-python3.9-standard:2024.02.1-b4' # Copy a runtime ID from previous output
createModelBuildResponse = deployment.createModelBuild(projectId, modelVersionId, modelCreationId, runtimeId)
modelBuildId = createModelBuildResponse.id

deployment.createModelDeployment(modelBuildId, projectId, modelCreationId)

## NOW TRY A REQUEST WITH THIS PAYLOAD!

#{"dataframe_split": {"columns": ["age", "credit_card_balance", "bank_account_balance", "mortgage_balance", "sec_bank_account_balance", "savings_account_balance", "sec_savings_account_balance", "total_est_nworth", "primary_loan_balance", "secondary_loan_balance", "uni_loan_balance", "longitude", "latitude", "transaction_amount"], "data":[[35.5, 20000.5, 3900.5, 14000.5, 2944.5, 3400.5, 12000.5, 29000.5, 1300.5, 15000.5, 10000.5, 2000.5, 90.5, 120.5]]}}
