import torch
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from datetime import datetime
from data_utils import Data_utils
from model import NN_model
from train import train_model
from predict import evaluate_model

def device_allocation():
    """Allocate device to cuda if it is available, otherwise allocate to CPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def data(batch_size):
    print("load and prepare the datasets")

    data = Data_utils()
    data.load_train_data(batch_size)
    data.load_test_data(batch_size)

    return data

def train(model, device, train_data, val_data, train_params):
    """ Put model to the device and train it"""
    model = model.to(device)
    model, history = train_model(model, train_data, val_data, train_params["epochs"], device, train_params["learning_rate"])
    return model, history

def predict(model, test_data, device):
    results = evaluate_model(model, test_data, device)
    return results

def plot_losses(history, results):
    fig, axs = plt.subplots(2,2)

    axs[0][0].plot(history["train_loss"], label='train')
    axs[0][0].plot(history["val_loss"], label='validation')
    axs[0][1].plot(results["test_loss"])

    axs[1][0].plot(history["train_acc"], label='train')
    axs[1][0].plot(history["val_acc"], label='validation')
    axs[1][1].plot(results["test_acc"])

    axs[0][0].set_xlabel("epochs")
    axs[1][0].set_xlabel("epochs")
    axs[0][1].set_xlabel("number of samples")
    axs[1][1].set_xlabel("number of samples")
    axs[0][0].set_ylabel("loss")
    axs[1][0].set_ylabel("loss")
    axs[0][1].set_ylabel("accuracy")
    axs[1][1].set_ylabel("accuracy")

    axs[0][0].set_title("Train and validation losses")
    axs[1][0].set_title("Train and validation accuracies")
    axs[0][1].set_title("Loss of the test set")
    axs[1][1].set_title("Accuracies of the test set")

    axs[0][0].legend()
    axs[1][0].legend()
    fig.set_size_inches(16.5, 16.5, forward=True)
    fig.tight_layout()

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    fig.savefig("results/graphs/curves_" + current_time +  ".jpg")
    plt.close(fig)

def hyper_tune_flow():

    device = device_allocation()
    epochs = 50
    batch_size = 8
        
    n_layers = 3
    learning_rate = 0.0001
    dimensions = [24, 16, 4]

    model_params = {
        "n_layers": int(n_layers),
        "dimensions": dimensions,
    }
    train_params = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
    }

    print("train model")

    all_data = data(train_params["batch_size"])
    model = NN_model(model_params)

    model, history = train(model, device, all_data.train_dataloader, all_data.val_dataloader, train_params)
    results = predict(model, all_data.test_dataloader, device)
    plot_losses(history, results)
    best_val = min(history["val_loss"])
    print("best validation loss: " + str(best_val)) 

if __name__ == '__main__':
    hyper_tune_flow()
        
