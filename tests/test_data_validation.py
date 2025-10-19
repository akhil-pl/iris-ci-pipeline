import pandas as pd
import pytest

DATA_FILES = [
    "data/iris_v1.csv",
    "data/iris_v2.csv"
]

EXPECTED_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

@pytest.mark.parametrize("file_path", DATA_FILES)
def test_csv_file_exists(file_path):
    """Check if CSV file exists and can be read."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        pytest.fail(f"{file_path} not found")
    except Exception as e:
        pytest.fail(f"Error reading {file_path}: {e}")

@pytest.mark.parametrize("file_path", DATA_FILES)
def test_csv_columns(file_path):
    """Check if CSV file has expected columns."""
    df = pd.read_csv(file_path)
    for col in EXPECTED_COLUMNS:
        assert col in df.columns, f"Column '{col}' not found in {file_path}"

@pytest.mark.parametrize("file_path", DATA_FILES)
def test_no_missing_values(file_path):
    """Check there are no missing values in the CSV."""
    df = pd.read_csv(file_path)
    assert df.isnull().sum().sum() == 0, f"Missing values found in {file_path}"

@pytest.mark.parametrize("file_path", DATA_FILES)
def test_column_types(file_path):
    """Check numeric columns are float and species is string."""
    df = pd.read_csv(file_path)
    numeric_cols = EXPECTED_COLUMNS[:-1]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} is not numeric"
    assert pd.api.types.is_object_dtype(df["species"]), "Column 'species' is not object/string type"
