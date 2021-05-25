# imports
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

class Trainer():
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
        return rmse


# if __name__ == "__main__":
#     # get data
#     # clean data
#     # set X and y
#     # hold out
#     print('TODO')
