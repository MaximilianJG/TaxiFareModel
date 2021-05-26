# imports
# from ml_flow_test import EXPERIMENT_NAME
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.params import BUCKET_NAME, LOCAL_MODEL_FILE, MODEL_STORAGE_LOCATION, REMOTE_MODEL_FILE, REQ_AI_PF_FILE, REQ_DESTINATION

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import joblib
from google.cloud import storage
import subprocess


class Trainer():
    
    MLFLOW_URI = "https://mlflow.lewagon.co/"
    EXPERIMENT_NAME = "[UK] [London] [MaximilianJG] TaxiFareModel + v1"
    
    def __init__(self, X, y, run_locally=False):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.run_locally = run_locally

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ("distance_transformer", DistanceTransformer()),
            ("standard_scaler", StandardScaler())
        ])
        time_pipe = Pipeline([
            ("time_encoder", TimeFeaturesEncoder('pickup_datetime')),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ])
        preproc_pipe = ColumnTransformer([
            ("distance", dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ("time", time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ("preproc", preproc_pipe),
            ("linear_model", LinearRegression())
        ])
        self.pipeline = pipe

    def run(self):
        """set and train the pipeline"""
        pipeline = self.set_pipeline()
        self.save_package_versions()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("rmse", rmse)
        return rmse
    
        
    def save_model(self):
        """ Save the trained model into a model.joblib file """
        if self.run_locally: 
            joblib.dump(self.pipeline, LOCAL_MODEL_FILE)
            print("saved local_model.joblib")
        else: 
            joblib.dump(self.pipeline, REMOTE_MODEL_FILE)
            print("saved remote_model.joblib")

        if not self.run_locally:
            self.upload_file_to_gcp(REMOTE_MODEL_FILE, MODEL_STORAGE_LOCATION, BUCKET_NAME)
            self.upload_file_to_gcp(REQ_AI_PF_FILE, REQ_DESTINATION, BUCKET_NAME)


        print(f"uploaded model.joblib to gcp cloud storage under \n => {MODEL_STORAGE_LOCATION}")


    def upload_file_to_gcp(self, source_file, destination_file, bucket_name): 
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_file)
        blob.upload_from_filename(source_file)
        

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri("https://mlflow.lewagon.co/")
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment("[UK] [London] [MaximilianJG] TaxiFareModel + v1")
        except BaseException:
            return self.mlflow_client.get_experiment_by_name("[UK] [London] [MaximilianJG] TaxiFareModel + v1").experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


    def save_package_versions(self):
        with open(REQ_AI_PF_FILE, 'w') as outfile:
            subprocess.call(["pip", "freeze"], stdout=outfile)
            

if __name__ == "__main__":
    df = get_data(nrows=1000)
    df = clean_data(df)
    X = df.drop(columns="fare_amount")
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    trainer = Trainer(X_train, y_train, run_locally=False) # change this with gcp_submit_training to False
    trainer.run()
    trainer.evaluate(X_test, y_test)
    trainer.save_model()
    
    # final_trainer = Trainer(X, y) # train on everything
    # y_pred = final_trainer.run().predict(X_test)
    