import argparse
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="A simple example of linear regression.")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
parser.add_argument("--seed", action="store", type=int, default=0,
                    help="Set the random seed for generating data")
parser.add_argument("--num-datapoints", action="store", type=int, default=100,
                    help="The number of datapoints")
parser.add_argument("--x-min", action="store", type=int, default=0,
                    help="Minimum x value for generating datapoints")
parser.add_argument("--x-max", action="store", type=int, default=100,
                    help="Maximum x value for generating datapoints")
parser.add_argument("--train-percent", action="store", type=float, default=50.0,
                    help="The percent of datpoints that will be used for training the model")
args = parser.parse_args()


def generate_random_datapoints(amount: int) -> tuple:
    x, y = [], []

    for i in range(0, amount):
        x.append(np.random.randint(args.x_min, args.x_max))
        y.append(np.random.randint(x[i], x[i] + 20))

    return (np.array(x), np.array(y))


if __name__ == "__main__":
    # Randomness seed used for generating datapoints
    np.random.seed(args.seed)

    # Generate the datapoints. x position and y position are stored in seperate arrays.
    x, y = generate_random_datapoints(args.num_datapoints)

    # We split our datapoints into two groups:
    # The `train_split` is used to train the model.
    # The `test_split` is what is used score the model's performance.
    train_split = int(len(x) * args.train_percent // 100)
    test_split = int(len(x) * (100 - args.train_percent) // 100)

    # `reshape` is just to convert `x_train` and `x_test` to 2D arrays (the model takes a 2D array).
    x_train = x[train_split:].reshape(-1, 1)
    y_train = y[train_split:]
    x_test = x[:test_split].reshape(-1, 1)
    y_test = y[:test_split]

    # Create and train the model.
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Have the model try to predict the y value of the test split.
    # This is the first time the model is seeing the testing set.
    y_pred = model.predict(x_test)

    if args.verbose:
        print("Datapoints generated: " + str(len(x)))
        print("Train/Test Splits (count): " + str(train_split) + "/" + str(test_split))
        print("Predicted line: y = " + str(model.coef_) + "x + " + str(model.intercept_))
        print("Actual line: y = x + 10")
        print("Accuracy: " + str(model.score(x_test, y_test)) + " (the closer to 1 the better)")

    # Plot the testing data and the line the model predicted.
    plt.scatter(x_test, y_test, color="black")
    plt.plot(x_test, y_pred, color="blue", linewidth=3)
    plt.show()