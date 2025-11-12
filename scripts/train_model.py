import mlflow
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

train_data_path = "data/yellow_tripdata_2023-01.parquet"
val_data_path = "data/yellow_tripdata_2023-02.parquet"
tracking_uri = "sqlite:///mlflow_32.db"
experiment_name = "Model-Management-Experiment"


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    return df


def compute_features(df):
    str_columns = ["PULocationID", "DOLocationID"]
    df[str_columns] = df[str_columns].astype("str")
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    return df


def extract_features_target(df):
    df = compute_features(df)

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    return df[categorical + numerical], df.duration.values


def main():
    print("Setting up MLflow...")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    df_train = read_data(train_data_path)
    df_val = read_data(val_data_path)

    X_train, y_train = extract_features_target(df_train)
    X_val, y_val = extract_features_target(df_val)

    # get columns as dict
    train_dicts = X_train.to_dict(orient="records")
    val_dicts = X_val.to_dict(orient="records")

    # fit transform dict vectorizer
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    mlflow.autolog(log_datasets=False)
    with mlflow.start_run():
        print("Training the model...")
        alpha = 0.6
        lr = Ridge(alpha)
        lr.fit(X_train, y_train)
        print("Evaluating the model...")
        y_train_pred = lr.predict(X_train)
        y_val_pred = lr.predict(X_val)
        print("Train RMSE:", mean_squared_error(y_train, y_train_pred))
        print("Validation RMSE:", mean_squared_error(y_val, y_val_pred))


if __name__ == "__main__":
    main()
