import argparse
import itertools
from typing import List, Tuple

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from scipy.sparse import csr_matrix

np.random.seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--penalty", choices=["l1","l2","elasticnet","none"], default="l2")
    parser.add_argument("-C", type=float, default=1.0)
    parser.add_argument("--solver", choices=["newton-cg","lbfgs","liblinear","sag","saga"], default="lbfgs")
    return parser.parse_args()

def prepare_data(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["sentiment"], random_state=1234)
    return train_df, test_df

def make_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[csr_matrix, csr_matrix]:
    vectorizer = TfidfVectorizer(stop_words="english")
    train_inputs = vectorizer.fit_transform(train_df["review"])
    test_inputs = vectorizer.transform(test_df["review"])
    return train_inputs, test_inputs

def train(train_inputs, train_outputs: np.ndarray, **model_kwargs) -> BaseEstimator:
    model = LogisticRegression(**model_kwargs)
    model.fit(train_inputs, train_outputs)
    return model

def evaluate(model: BaseEstimator, test_inputs: csr_matrix, test_outputs: np.ndarray, class_names: List[str]) -> Tuple[float, Figure]:
    predicted_test_outputs = model.predict(test_inputs)
    return f1_score(test_outputs, predicted_test_outputs)

def main(args):
    #Tracking URI added here
    mlflow.set_tracking_uri(uri="http://localhost:6101")
    
    df = pd.read_csv("./imdbdataset.csv")
    df["label"] = pd.factorize(df["sentiment"])[0]

    test_size = 0.3
    train_df, test_df = prepare_data(df, test_size=test_size)

    mlflow.set_experiment("tracking-demo")
    with mlflow.start_run():
        train_inputs, test_inputs = make_features(train_df, test_df)
        model = train(
            train_inputs,
            train_df["label"].values,
            penalty=args.penalty,
            C=args.C,
            solver=args.solver
        )
        f1_score = evaluate(model, test_inputs, test_df["label"].values, df["sentiment"].unique().tolist())
        print("F1 Score:", f1_score)

        mlflow.log_param("test_size", test_size)
        mlflow.log_param("C", args.C)
        mlflow.log_metric("f1_score", f1_score)
        #mlflow.log_figure(figure, "figure.png")

if __name__ == "__main__":
    main(parse_args())
