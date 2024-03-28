# Logging Iceberg Metrics with MLFlow Tracking in CML

Cloudera Machine Learning (CML) on Cloudera Data Platform (CDP) accelerates time-to-value by enabling data scientists to collaborate in a single unified platform that is all inclusive for powering any AI use case. Purpose-built for agile experimentation and production ML workflows, Cloudera Machine Learning manages everything from data preparation to MLOps, to predictive reporting.

CML is compatible with the MLflow Tracking API and makes use of the MLflow client library as the default method to log experiments. Existing projects with existing experiments are still available and usable. CML’s experiment tracking features allow you to use the MLflow client library for logging parameters, code versions, metrics, and output files when running your machine learning code.

Apache Iceberg is a table format for huge analytics datasets in the cloud that defines how metadata is stored and data files are organized. Iceberg is also a library that compute engines can use to read/write a table. CML offers Data Connections to connect to Data Sources available within the CDP Environment including Iceberg Open Lakehouses.

In this example we will create an experiment with MLFlow Tracking and log Iceberg metadata in order to enhance machine learning reproducibility in the context of MLOps.

## Step by Step Guide

The code samples provided below are extracts from the accompanying notebook. The full code can be found in this [git repository]().

#### Setup

Create a CML Project with Python 3.9 / JupyterLab Editor Runtime.
Launch a CML Session and install requirements.

#### Run Notebook

Run each cell in the notebook.

#### Code highlights:

* MLFlow Tracking supports modules built specifically for some of the most popular open source frameworks. In this case we will import "mlflow.spark"
* You can leverage CML Spark Data Connections to launch a SparkSession object with the recommended Iceberg Spark configurations. Spark Data Connections make connecting to your Iceberg data effortless.

```
import mlflow.spark
import cml.data_v1 as cmldata

#Edit your connection name here:
CONNECTION_NAME = "se-aw-mdl"

conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()
```

* The exp1 method acts as a wrapper to your first MLFlow experiment.
* The experiment name is set with the mlflow.set_exeperiment method.
* Data is written from a PySpark dataframe to an Iceberg table via a simple routine: "df.writeTo().createOrReplace()"
* Iceberg History and Snapshots tables are available for you to monitor Iceberg metadata. In this example we save the latest snapshot ID along with its timestamp and parent snapshot ID into Python variables.
* Within the context of this experiment run, a Spark ML Pipeline is trained to tokenize and classify text.
* MLFlow Tracking allows you to set custom tags. These tags can be used to search your experiments using the MLFlow client.
* MLFlow Tracking allows you to create a run context to track metrics according to a specific run. In this particular case we use log_metric method to track the Iceberg variables corresponding to snaphot and write operation timestamp.   
* Once the experiment completes you can retrieve its ID and more metadata using the MLFlow client.

```
def exp1(df):

    mlflow.set_experiment("sparkml-experiment")

    ##EXPERIMENT 1

    df.writeTo("spark_catalog.default.training").using("iceberg").createOrReplace()
    spark.sql("SELECT * FROM spark_catalog.default.training").show()

    ### SHOW TABLE HISTORY AND SNAPSHOTS
    spark.read.format("iceberg").load("spark_catalog.default.training.history").show(20, False)
    spark.read.format("iceberg").load("spark_catalog.default.training.snapshots").show(20, False)

    snapshot_id = spark.read.format("iceberg").load("spark_catalog.default.training.snapshots").select("snapshot_id").tail(1)[0][0]
    committed_at = spark.read.format("iceberg").load("spark_catalog.default.training.snapshots").select("committed_at").tail(1)[0][0].strftime('%m/%d/%Y')
    parent_id = spark.read.format("iceberg").load("spark_catalog.default.training.snapshots").select("parent_id").tail(1)[0][0]

    tags = {
      "iceberg_snapshot_id": snapshot_id,
      "iceberg_snapshot_committed_at": committed_at,
      "iceberg_parent_id": parent_id,
      "row_count": training_df.count()
    }

    ### MLFLOW EXPERIMENT RUN
    with mlflow.start_run() as run:

        maxIter=8
        regParam=0.01

        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
        lr = LogisticRegression(maxIter=maxIter, regParam=regParam)
        pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
        model = pipeline.fit(training_df)

        mlflow.log_param("maxIter", maxIter)
        mlflow.log_param("regParam", regParam)

        #prediction = model.transform(test)
        mlflow.set_tags(tags)

    mlflow.end_run()

    experiment_id = mlflow.get_experiment_by_name("sparkml-experiment").experiment_id
    runs_df = mlflow.search_runs(experiment_id, run_view_type=1)

    return runs_df
```

* The second experiment is very similar to the first, except data is appended to the Iceberg table via "df.writeTo().append()"
* As data is inserted into the table, new Iceberg metadata is generated in the Iceberg Metadata Layer and becomes available in the Snapshots and History tables. This metadata is tracked into new Python variables.
* In this particular example we again use the "log_metric" method to track the Iceberg Snapshot ID and Timestamp for this append operation.
* Within the context of this experiment run, the Spark ML Pipeline is retrained for the same purpose of tokenizing and classifying text, but using the new version of the data after the append operation.

```
def exp2(df):

    mlflow.set_experiment("sparkml-experiment")

    ##EXPERIMENT 2

    ### ICEBERG INSERT DATA - APPEND FROM DATAFRAME
    # PRE-INSERT
    spark.sql("SELECT * FROM spark_catalog.default.training").show()

    temp_df = spark.sql("SELECT * FROM spark_catalog.default.training")
    temp_df.writeTo("spark_catalog.default.training").append()
    df = spark.sql("SELECT * FROM spark_catalog.default.training")

    # PROST-INSERT
    spark.sql("SELECT * FROM spark_catalog.default.training").show()

    spark.read.format("iceberg").load("spark_catalog.default.training.history").show(20, False)
    spark.read.format("iceberg").load("spark_catalog.default.training.snapshots").show(20, False)

    snapshot_id = spark.read.format("iceberg").load("spark_catalog.default.training.snapshots").select("snapshot_id").tail(1)[0][0]
    committed_at = spark.read.format("iceberg").load("spark_catalog.default.training.snapshots").select("committed_at").tail(1)[0][0].strftime('%m/%d/%Y')
    parent_id = spark.read.format("iceberg").load("spark_catalog.default.training.snapshots").select("parent_id").tail(1)[0][0]

    tags = {
      "iceberg_snapshot_id": snapshot_id,
      "iceberg_snapshot_committed_at": committed_at,
      "iceberg_parent_id": parent_id,
      "row_count": df.count()
    }

    ### MLFLOW EXPERIMENT RUN
    with mlflow.start_run() as run:

        maxIter=10
        regParam=0.002

        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
        lr = LogisticRegression(maxIter=maxIter, regParam=regParam)
        pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
        model = pipeline.fit(training_df)

        mlflow.log_param("maxIter", maxIter)
        mlflow.log_param("regParam", regParam)

        #prediction = model.transform(test)
        mlflow.set_tags(tags)

    mlflow.end_run()

    experiment_id = mlflow.get_experiment_by_name("sparkml-experiment").experiment_id
    runs_df = mlflow.search_runs(experiment_id, run_view_type=1)

    return runs_df
```

* Finally, in the third experiment we retrain the Spark ML Pipeline but first we retrieve the data as it was prior to the append operation by applying the provided Iceberg Snapshot ID in the "spark.read.table" method.

```
def exp3(df, snapshot_id):
    ##EXPERIMENT 3

    df = spark.read.option("snapshot-id", snapshot_id).table("spark_catalog.default.training")

    committed_at = spark.sql("SELECT committed_at FROM spark_catalog.default.training.snapshots WHERE snapshot_id = {};".format(snapshot_id)).collect()[0][0].strftime('%m/%d/%Y')
    parent_id = str(spark.sql("SELECT parent_id FROM spark_catalog.default.training.snapshots WHERE snapshot_id = {};".format(snapshot_id)).tail(1)[0][0])

    tags = {
      "iceberg_snapshot_id": snapshot_id,
      "iceberg_snapshot_committed_at": committed_at,
      "iceberg_parent_id": parent_id,
      "row_count": training_df.count()
    }

    ### MLFLOW EXPERIMENT RUN
    with mlflow.start_run() as run:

        maxIter=7
        regParam=0.005

        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
        lr = LogisticRegression(maxIter=maxIter, regParam=regParam)
        pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
        model = pipeline.fit(training_df)

        mlflow.log_param("maxIter", maxIter)
        mlflow.log_param("regParam", regParam)

        #prediction = model.transform(test)
        mlflow.set_tags(tags)

    mlflow.end_run()

    experiment_id = mlflow.get_experiment_by_name("sparkml-experiment").experiment_id
    runs_df = mlflow.search_runs(experiment_id, run_view_type=1)

    #spark.stop()

    return runs_df
```

## Summary and Next Steps

Large ML organizations require standardized best practices such as tracking models and respective dependencies, model developers, and matching those with datasets in order to keep a consistent view of all MLOps practices.

MLFlow Tracking in CML allows you to achieve this goal by allowing you to specify datasets and other custom metadata when tracking experiment runs. In the above example we tracked Iceberg metadata in order to allow data scientists to retrain an existing pipeline with datasets as of arbitrary points in time. In the process, we used tags in order to implement a consistent taxonomy across all experiment runs.

* [CML Model Deployment with MLFlow and APIv2](https://community.cloudera.com/t5/Community-Articles/CML-Model-Deployment-with-MLFlow-and-APIv2/ta-p/385656)
* [Spark in CML: Recommendatons for using Spark](https://community.cloudera.com/t5/Community-Articles/Spark-in-CML-Recommendations-for-using-Spark-in-Cloudera/ta-p/372164#:~:text=Spark%20Dynamic%20Allocation%20is%20enabled,order%20to%20prevent%20runaway%20charges.)
* [Experiments with MLFlow](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-experiments-v2.html)
* [Registering and Deploying Models with Model Registry](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-registering-and-deploying-a-model-registry.html)
* [Apache Iceberg Documentation](https://iceberg.apache.org/docs/latest/)
* [Iceberg Time Travel](https://iceberg.apache.org/docs/latest/spark-queries/#time-travel)
* [Introducing MLOps and SDX for Models in CML](https://blog.cloudera.com/introducing-mlops-and-sdx-for-models-in-cloudera-machine-learning/)
