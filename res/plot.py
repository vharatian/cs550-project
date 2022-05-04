import matplotlib.pyplot as plt
from data_exp_1 import *
from data_exp_2 import *
import numpy as np
import math

PLOT_DIR = "../img/"
FUNC_SEPARATOR = "=========================================="
EXP_SEPARATOR = "***********"

def num_hidden_layers():
    FUNC_NAME = "num_hidden_layers"
    data_arr = [data_exp_1_arr, data_exp_2_arr]

    for data in data_arr:
        np_arr = np.array(data)
        target_num_hidden_layers = num_hidden_layers_exp_2 if data_arr.index(data)==1 else num_hidden_layers_exp_1
        for num_hidden_layers in target_num_hidden_layers:
            y_displayed = np_arr[np_arr[:,1] == num_hidden_layers] # num_hidden_layers index is 5
            print(y_displayed)
            y_axis = y_displayed[:,[0]] # accuracy is at index 0
            y_axis = y_axis.flatten().astype(np.float)*100
            labels = np.arange(1,len(y_axis) + 1, 1)
            plt.plot(labels, y_axis, linewidth=2, label=num_hidden_layers)
        plt.yticks(np.arange(math.floor(min(y_axis))-2, math.ceil(max(y_axis))+2, step=
                             1.5))
        plt.xticks(np.arange(1,len(y_axis) + 1, 5))
        plt.legend(loc="lower right")
        plt.title("num_hidden_layers Sweep")
        plt.xlabel("Configuration Number")
        plt.ylabel("Accuracy (%)")
        plt.grid()
        plt.savefig((PLOT_DIR+FUNC_NAME+str(data_arr.index(data))+'.jpg'), dpi=300)
        plt.close()
        print(EXP_SEPARATOR)

def epoch():
    FUNC_NAME = "epoch"
    print(FUNC_NAME)
    data_arr = [data_exp_1_arr, data_exp_2_arr]

    for data in data_arr:
        np_arr = np.array(data)
        target_epoch = epochs_exp_2 if data_arr.index(data)==1 else epochs_exp_1
        for epoch in target_epoch:
            y_displayed = np_arr[np_arr[:,5] == epoch] # epoch index is 5
            print(y_displayed)
            y_axis = y_displayed[:,[0]] # accuracy is at index 0
            y_axis = y_axis.flatten().astype(np.float)*100
            labels = np.arange(1,len(y_axis) + 1, 1)
            plt.plot(labels, y_axis, linewidth=2, label=epoch)
        plt.yticks(np.arange(math.floor(min(y_axis))-2, math.ceil(max(y_axis))+2, step=
                             1.5))
        plt.xticks(np.arange(1,len(y_axis) + 1, 5))
        plt.legend(loc="lower right")
        plt.title("Epoch Sweep")
        plt.xlabel("Configuration Number")
        plt.ylabel("Accuracy (%)")
        plt.grid()
        plt.savefig((PLOT_DIR+FUNC_NAME+str(data_arr.index(data))+'.jpg'), dpi=300)
        plt.close()
        print(EXP_SEPARATOR)

def num_hidden_units():
    FUNC_NAME = "num_hidden_units"
    print(FUNC_NAME)
    data_arr = [data_exp_1_arr, data_exp_2_arr]

    for data in data_arr:
        np_arr = np.array(data)
        target_num_hidden_units = num_hidden_units_exp_2 if data_arr.index(data)==1 else num_hidden_units_exp_1
        for num_hidden_units in target_num_hidden_units:
            y_displayed = np_arr[np_arr[:,2] == num_hidden_units] # num_hidden_units index is 5
            print(y_displayed)
            y_axis = y_displayed[:,[0]] # accuracy is at index 0
            y_axis = y_axis.flatten().astype(np.float)*100
            labels = np.arange(1,len(y_axis) + 1, 1)
            plt.plot(labels, y_axis, linewidth=2, label=num_hidden_units)
        plt.yticks(np.arange(math.floor(min(y_axis))-2, math.ceil(max(y_axis))+2, step=
                             1.5))
        plt.xticks(np.arange(1,len(y_axis) + 1, 5))
        plt.legend(loc="lower right")
        plt.title("num_hidden_units Sweep")
        plt.xlabel("Configuration Number")
        plt.ylabel("Accuracy (%)")
        plt.grid()
        plt.savefig((PLOT_DIR+FUNC_NAME+str(data_arr.index(data))+'.jpg'), dpi=300)
        plt.close()
        print(EXP_SEPARATOR)

def learning_rate():
    FUNC_NAME = "learning_rate"
    print(FUNC_NAME)
    data_arr = [data_exp_1_arr, data_exp_2_arr]

    for data in data_arr:
        np_arr = np.array(data)
        target_learning_rate = lr_exp_2 if data_arr.index(data)==1 else lr_exp_1
        for learning_rate in target_learning_rate:
            y_displayed = np_arr[np_arr[:,3] == learning_rate] # learning_rate index is 5
            print(y_displayed)
            y_axis = y_displayed[:,[0]] # accuracy is at index 0
            y_axis = y_axis.flatten().astype(np.float)*100
            labels = np.arange(1,len(y_axis) + 1, 1)
            plt.plot(labels, y_axis, linewidth=2, label=learning_rate)
        plt.yticks(np.arange(math.floor(min(y_axis))-2, math.ceil(max(y_axis))+2, step=
                             1.5))
        plt.xticks(np.arange(1,len(y_axis) + 1, 5))
        plt.legend(loc="lower right")
        plt.title("learning_rate Sweep")
        plt.xlabel("Configuration Number")
        plt.ylabel("Accuracy (%)")
        plt.grid()
        plt.savefig((PLOT_DIR+FUNC_NAME+str(data_arr.index(data))+'.jpg'), dpi=300)
        plt.close()
        print(EXP_SEPARATOR)

def initialization():
    FUNC_NAME = "initialization"
    print(FUNC_NAME)
    data_arr = [data_exp_1_arr, data_exp_2_arr]

    for data in data_arr:
        np_arr = np.array(data)
        target_initialization = initial_weights_exp_2 if data_arr.index(data)==1 else initial_weights_exp_1
        for Initialization in target_initialization:
            y_displayed = np_arr[np_arr[:,4] == Initialization] # Initialization index is 5
            print(y_displayed)
            y_axis = y_displayed[:,[0]] # accuracy is at index 0
            y_axis = y_axis.flatten().astype(np.float)*100
            labels = np.arange(1,len(y_axis) + 1, 1)
            plt.plot(labels, y_axis, linewidth=2, label=Initialization)
        plt.yticks(np.arange(math.floor(min(y_axis))-2, math.ceil(max(y_axis))+2, step=
                             1.5))
        plt.xticks(np.arange(1,len(y_axis) + 1, 5))
        plt.legend(loc="lower right")
        plt.title("Initialization Sweep")
        plt.xlabel("Configuration Number")
        plt.ylabel("Accuracy (%)")
        plt.grid()
        plt.savefig((PLOT_DIR+FUNC_NAME+str(data_arr.index(data))+'.jpg'), dpi=300)
        plt.close()
        print(EXP_SEPARATOR)

def main():
    print(FUNC_SEPARATOR)
    epoch()
    print(FUNC_SEPARATOR)
    print(" ")
    print(FUNC_SEPARATOR)
    initialization()
    print(FUNC_SEPARATOR)
    print(" ")
    print(FUNC_SEPARATOR)
    learning_rate()
    print(FUNC_SEPARATOR)
    print(" ")
    print(FUNC_SEPARATOR)
    num_hidden_units()
    print(FUNC_SEPARATOR)
    print(" ")
    print(FUNC_SEPARATOR)
    num_hidden_layers()
    print(FUNC_SEPARATOR)
    print(" ")
    print(FUNC_SEPARATOR)

if __name__ == '__main__':
    main()
