from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


def get_raw_data():
    """Loads the Iris dataset and saves it to a raw data file."""
    print("Fetching raw data...")
    # Define file path
    data_path = Path("data/raw")
    data_path.mkdir(parents=True, exist_ok=True)
    iris_csv_path = data_path / "iris.csv"

    # Load dataset
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["target"] = iris.target

    # Save to CSV
    iris_df.to_csv(iris_csv_path, index=False)
    print(f"Raw data saved to {iris_csv_path}")


if __name__ == "__main__":
    get_raw_data()
