def create_train_test(csv_dataset):
    """
    Create 75% train and 25% test datasets from the given CSV dataset
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Load the dataset from the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_dataset)

    # Split the DataFrame into features (X) and class/label (y)
    X = df.drop(columns=['class', 'label']).to_numpy()
    y = df.label.to_numpy()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test