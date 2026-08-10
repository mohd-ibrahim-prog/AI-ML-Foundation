from sklearn.linear_model import LinearRegression


def train_model(X_train, y_train):
    """Train a linear regression model using the training data."""

    model = LinearRegression()

    model.fit(X_train, y_train)

    print("\nLinear Regression model trained successfully.")

    print("\nModel Coefficients")
    print("------------------")

    for feature, coefficient in zip(X_train.columns, model.coef_):
        print(f"{feature:<20}: {coefficient:.4f}")

    print(f"\nIntercept             : {model.intercept_:.4f}")

    return model