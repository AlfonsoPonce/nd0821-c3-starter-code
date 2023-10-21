# Script to train machine learning model.
import os.path

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
import logging


# Initialize logging
logging.basicConfig(filename='logs.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')
# Add the necessary imports for the starter code.


logging.info("Reading Data")
data = pd.read_csv(str(Path('../data/cleaned_data.csv')))

logging.info("Splitting data")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

logging.info("Processing categorical columns")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

logging.info("Training model")
# Train and save a model.
model = train_model(X_train, y_train)

logging.info("Saving model, encoder and labeler")
save_folder= '../model/'
joblib.dump(model, os.path.join(save_folder, 'trained_model.pkl'))
joblib.dump(encoder, os.path.join(save_folder, 'encoder.pkl'))
joblib.dump(lb, os.path.join(save_folder, 'lb.pkl'))


