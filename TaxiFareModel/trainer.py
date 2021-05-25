# imports
from ml_flow_test import EXPERIMENT_NAME
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split


class Trainer():
    
    MLFLOW_URI = "https://mlflow.lewagon.co/"
    EXPERIMENT_NAME = "[UK] [London] [MaximilianJG] TaxiFareModel + v1"
    
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

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
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("rmse", rmse)
        return rmse
    
    
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


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    X = df.drop(columns="fare_amount")
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    trainer = Trainer(X, y)
    trainer.run()
    trainer.evaluate(X_test, y_test)
