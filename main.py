import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from scipy.stats import zscore

np.random.seed(42)


def get_compiled_model(experiment):
    model = Sequential()
    for _ in range(experiment.hidden_layer_count):
        model.add(Dense(experiment.hidden_layer_units,
                        activation=experiment.activation_method,
                        kernel_initializer=experiment.initial_weights))

    model.add(Dense(1,
                    activation='sigmoid',
                    kernel_initializer=experiment.initial_weights))

    if experiment.optimizer_method == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif experiment.optimizer_method == "sgd":
        opt = keras.optimizers.SGD(learning_rate=lr)

    model.compile(
        loss='binary_crossentropy',
        optimizer=opt)

    return model


def cross_val(experiment, X, y):
    size = int(len(X) / experiment.folding_size)
    total_acc = 0
    for i in range(experiment.folding_size):
        begin = size * i
        end = size * (i + 1)
        if i == experiment.folding_size - 1:
            end = len(X)

        X_val = X[begin:end]
        y_val = y[begin:end]
        X_train = np.concatenate((X[:begin], X[end:]), axis=0)
        y_train = np.concatenate((y[:begin], y[end:]), axis=0)

        model = get_compiled_model(experiment)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=experiment.epoch, verbose=0)
        y_pred = (model.predict(X_val) > 0.5).astype("int32")
        total_acc += accuracy_score(y_val, y_pred)

    avg_acc = total_acc / experiment.folding_size
    return ExperimentResult(experiment, avg_acc)


def preprocess_bsl(df):
    df = df.drop("SHA", axis=1)
    y = df.pop('defect').to_numpy().astype(int)
    df = df.apply(zscore)
    X = df.to_numpy().astype("float32")

    return X, y


class Experiment:
    def __init__(self, folding_size, hidden_layer_count, hidden_layer_units, activation_method, optimizer_method,
                 learning_rate, initial_weights, epoch):
        self.initial_weights = initial_weights
        self.optimizer_method = optimizer_method
        self.activation_method = activation_method
        self.folding_size = folding_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.hidden_layer_units = hidden_layer_units
        self.hidden_layer_count = hidden_layer_count

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    def __str__(self):
        return self.toJson()

    def __repr__(self):
        return f"{self}"


class ExperimentResult:
    def __init__(self, experiment, accuracy):
        self.experiment = experiment
        self.accuracy = accuracy

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    def __str__(self):
        return self.toJson()

    def __repr__(self):
        return f"{self}"


class ExperimentResultGroup:
    def __init__(self, group_name, results):
        self.results = results
        self.group_name = group_name


def group_results(results, variable_name):
    groups = []
    for r in results:
        variable_value = getattr(r.experiment, variable_name)
        group = None
        for g in groups:
            if g.group_name == variable_value:
                group = g
                break

        if group is None:
            group = ExperimentResultGroup(variable_value, [])
            groups += [group]

        group.results += [r]

    return groups




df = pd.read_csv('Datasets/baseline.csv')
X, y = preprocess_bsl(df)

best_acc = 0
results = []
# for num_hidden_layers in [1, 2, 3, 4]:
for num_hidden_layers in [1]:
    # for num_hidden_units in [1, 2, 4, 8]:
    for num_hidden_units in [1]:
        for lr in [0.1, 0.01, 0.001]:
        # for lr in [0.1]:
            for initial_weights in ["glorot_normal"]:
                # for epochs in [200, 500]:
                for epochs in [1, 2]:
                    experiment = Experiment(50, num_hidden_layers, num_hidden_units, "relu", "adam", lr,
                                            initial_weights, epochs)
                    print(experiment)
                    result = cross_val(experiment, X, y)
                    results.append(result)


# results = [
#     ExperimentResult(Experiment(1, None, None, None, None, None, None, 1), 0),
#     ExperimentResult(Experiment(1, None, None, None, None, None, None, 2), 0.1),
#     ExperimentResult(Experiment(1, None, None, None, None, None, None, 3), 0.2),
#     ExperimentResult(Experiment(2, None, None, None, None, None, None, 1), 0.1),
#     ExperimentResult(Experiment(2, None, None, None, None, None, None, 2), 0.2),
#     ExperimentResult(Experiment(2, None, None, None, None, None, None, 3), 0.3),
#     ExperimentResult(Experiment(3, None, None, None, None, None, None, 1), 0.2),
#     ExperimentResult(Experiment(3, None, None, None, None, None, None, 2), 0.3),
#     ExperimentResult(Experiment(3, None, None, None, None, None, None, 3), 0.4),
# ]

results.sort(key=lambda x: x.accuracy)
plot_variables = ["learning_rate"]
for pv in plot_variables:
    groups = group_results(results, pv)
    plt.title(pv)
    plt.xlabel("Configurations")
    plt.ylabel("Accuracy")

    for g in groups:
        x = list(range(len(g.results)))
        y = [r.accuracy for r in g.results]
        plt.plot(x, y, label=g.group_name)

    plt.legend(loc='best')
    plt.savefig(f"plots/{pv}.png")
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()

# print(results)
