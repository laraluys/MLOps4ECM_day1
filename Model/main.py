import torch
import numpy as np
import os
import mlflow
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from datetime import datetime
from data_utils import Data_utils
from model import NN_model
from train import train_model
from predict import evaluate_model

def champion_callback(study, frozen_trial):
    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


# @task(name="DEVICE", retries=0, retry_delay_seconds=15)
def device_allocation():
    """Allocate device to cuda if it is available, otherwise allocate to CPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

# @task(name="LOAD_DATA", retries=0, retry_delay_seconds=15)
def data(batch_size):
    print("load and prepare the datasets")

    data = Data_utils()
    data.load_train_data(batch_size)
    data.load_test_data(batch_size)

    return data

def train(model, device, train_data, val_data, train_params, trial):
    """ Put model to the device and train it"""
    model = model.to(device)
    model, history = train_model(model, train_data, val_data, train_params["epochs"], device, train_params["learning_rate"], trial)
    return model, history

def predict(model, test_data, device):
    results = evaluate_model(model, test_data, device)
    return results

def save_trial(trial:optuna.trial.Trial, objective):
    number = str(trial.number)
    text = str(trial.params)
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open("results/trial_values.txt", "a") as f:
        f.write(str(number) + "-" + str(text) + " " + str(current_time) + " " + str(objective) + '\n')

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
    # fig.savefig("models/graphs/input_vs_output/" + str(model_name) + "_" + str(type_layers) + "/curves_" + current_time + "_" + str(model_name) + "_" + str(type_layers) +  ".jpg")
    fig.savefig("results/graphs/curves_" + current_time +  ".jpg")
    plt.close(fig)

def hyper_tune_flow(trial:optuna.trial):
    mlflow.pytorch.autolog(disable=True)
    with mlflow.start_run(nested=True, log_system_metrics=False):
        # fixed variables
        device = device_allocation()
        epochs = 60
        batch_size = 8
            
        # dynamic varibales, chosen by optuna
        n_layers = trial.suggest_int("n_layers", 2, 10)  
        learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.001)
        dimensions = []
        for i in range(n_layers):
            embedding = trial.suggest_int(f"hidden_dim_{i}", 2, 99)
            dimensions.append(embedding)

        # save all parameters in dictionaries
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
        try:
            all_data = data(train_params["batch_size"])
            model = NN_model(model_params)

            # set up mlflow experiment
            mlflow.log_params(train_params)
            mlflow.log_params(model_params)
            model, history = train(model, device, all_data.train_dataloader, all_data.val_dataloader, train_params, trial)
            results = predict(model, all_data.test_dataloader, device)
            metrics_test = {
                "acc_test": np.mean(results["test_loss"]),
                "loss_test": np.mean(results["test_acc"])
            }
            mlflow.log_metrics(metrics_test)
            plot_losses(history, results)
            best_val = min(history["val_loss"])
            save_trial(trial, best_val)

        except ValueError as e:
            error = f"Error encountered in training: {e}. Pruning this trial."
            print(error)
            save_trial(trial, error)
            raise optuna.exceptions.TrialPruned()
    return best_val   

def get_or_create_experiment(experiment_name:str):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
      return experiment.experiment_id
    else:
      return mlflow.create_experiment(experiment_name)


if __name__ == '__main__':
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    exp_id = get_or_create_experiment("MLOps4ECM")
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run(experiment_id=exp_id, run_name="Neural_Network", nested=True):
        study_name = "Neural_Network"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize", 
            sampler = optuna.samplers.TPESampler()
            )
        study.optimize(hyper_tune_flow, n_trials=3000, callbacks=[champion_callback] )
        best_trial = study.best_trial
        best_params = {}
        dimensions = []
        print("best trial: " + str(best_trial.value))
        for key, value in best_trial.params.items():
            print("     {}: {}".format(key, value))
            if key == "n_layers":
                best_params["n_layers"] = value
            if "hidden_dim" in key:
                dimensions.append(value)
        best_params["dimensions"] = dimensions
        mlflow.log_metric("best_loss", study.best_value)
        mlflow.log_params(study.best_params)
        fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig2 = optuna.visualization.matplotlib.plot_param_importances(study)

        mlflow.log_figure(figure=fig1.figure, artifact_file="optimization_history.png")
        mlflow.log_figure(figure=fig2.figure, artifact_file="parameter_importance.png")
        save_dir = "model_params/" 

        model = NN_model(best_params)
        best_model_file = os.path.join(save_dir, '{}-{:.4f}-full_model.pth'.format(best_trial._trial_id, study.best_value))
        model.load_state_dict(torch.load(best_model_file, weights_only=True))
        mlflow.pytorch.log_model(model, "model")
        artifact_path = "model"
        model_uri = mlflow.get_artifact_uri(artifact_path)
