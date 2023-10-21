import pandas as pd
import pytest
import logging

@pytest.fixture(scope='module')
def data():
    return pd.read_csv('data/cleaned_data.csv')

@pytest.fixture(scope="module")
def path():
    return "./data/cleaned_census.csv"


def test_model_columns(data):
    '''
    Checks if data columns don't have trailing whitespaces
    '''
    try:
        for c in data.columns:
            assert (c[0] != ' ' or c[-1] != ' ')
    except AssertionError as err:
        logging.error("Trailing Whitespaces not removed")
        raise err

def test_data_size(data):
    '''
    Checks if there is a reasonable number of instances
    '''
    try:
        assert len(data[0]) > 1000 and len(data[0]) < 10000000
    except AssertionError as err:
        logging.error("Data size is not suitable")
        raise err

def test_import_data(path):
    """
    Test presence and shape of dataset file
    """
    try:
        df = pd.read_csv(path)

    except FileNotFoundError as err:
        logging.error("File not found")
        raise err


    # Check the df shape
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
