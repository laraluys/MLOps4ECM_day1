# MLflow implementation

MLflow is a tool used for experiment tracking. It can keep track of your models, its hyperparameters and its metrics, so you can have an overview of all your experiments in one place. We again start in the main function.

## imports

```
    import mlflow
    import mlflow.pytorch
```

## Set up mlflow experiment

The first thing you need to do for mlflow is to set a tracking uri. This should be the uri of where your mlflow server will live. Right now our server will run on the local host.

```
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
```

Now that mlflow knows where to send its information we can start an experiment. For this we create a seperate function which checks if an experiment is already with the given name. If not, a new experiment is created. Otherwise, the experiment with the given name is used.

```
def get_or_create_experiment(experiment_name:str):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
      return experiment.experiment_id
    else:
      return mlflow.create_experiment(experiment_name)

# in our objective function
exp_id = get_or_create_experiment("experiment_name")
mlflow.set_experiment(experiment_id=exp_id)
```

MLflow does have an autolog function for certain types of models. However, right now we want to do our own setup. Therefore we disable the autologging for pytorch.

```
mlflow.pytorch.autolog(disable=True)
```
Now we can start with the actual logging of the different variables

## logging variable

To keep track of the different variables we need to start a run. 

```
 
 with mlflow.start_run(run_name = "run_name"):
        // parameter logging
        // model training
        // model evaluation
        // metric logging
```
Everything we want to log has to be included in this run. This includes the training loop and model evaluation.

Logging the different parameters of your model is pretty easy to do. You just need to make a library of the parameters and then log this to mlflow using the log_params() function.

```
parameters = {
        "batch_size": 20,
        "epochs": 600,
        "learning_rate": 0.15
    }
mlflow.log_params(parameters)
```

Once you have trained and evaluated your model you might want to save the results or metrics, for example your minimum validation loss. This is done in a similar way as saving the model parameters.

```
min_val_loss = train_model(model, data)
accuracy = evaluate_model(model, data)
metrics = {
        "model_accuracy": accuracy,
        "minimum_validation_loss": min_val_loss,
    }
mlflow.log_metrics(metrics)
```

There might be some parameters that you would want to keep track of during the training loop to be able to create graphs in mlflow. For this you can log metrics during the training loop as well. The following code can be used in the train.py file.

```
mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)
```
Like before we create a dictionary with the metrics we want to save and then we use the log_metrics function. However this time we add the epoch as the step. This way the metric will be saved at each epoch, which will create a graph of the training and validation loss over the training time in mlflow.

## log artifacts

It might be that you want to save a figure that corresponds with your model. This can be a graph or for example a confusion matrix. This can again be easily done using the log_figure() method.

```
fig = plot_figure(data)
mlflow.log_figure(figure=fig, artifact_file="figure_name.png")
```

The artifact file is the file that the artifact (here a figure) well be saved in mlflow. In this case the figure is saved as an png file. Whatch out, mlflow cannot save .html files.

Lastly, you migth also want to save the model itself. This is done using the log_model function.

```
mlflow.pytorch.log_model(model, "model_name")
artifact_path = "model"
model_uri = mlflow.get_artifact_uri(artifact_path)
```
The last two lines of code save the model to a specific directory in the MLflow experiment.

## The MLflow server

Before we can run our code, our MLflow server needs to be up and running. This can very easily be done in the terminal with the following command.

```
mlflow server --host 127.0.0.1 --port 8080
```
As mentioned before, the host of our server right now is the localhost and the port we use is 8080. This should match the variable ins the set_tracking_uri() function.

When we run our code, we can go this uri and view our experiments.
![Mlflow_server_experiment_screenshot](images/mlflow_first.png)

As seen in the image we get an overview of all our different experiments on the left side. When you click on one of your experiments, you find the different runs. You can also see if the run has succesfully finished or if something went wrong on the way (green checkmark vs. red cross).

If you den click on one of the different runs you get the following screen. This includes all the information about your run.
![Mlflow_server_run_screenshot](images/mlflow_second.png)

When you scroll down, you see the different metrics and parameters that you saved.

![MLflow_server_param_metrics](images/mlflow_third.png)

If you then go to the model metrics folder at the top, you get an overview of your saved metrics, including the graphs of the metrics that were saved during training.

![MLflow_metrics_over_time](images/mlflow_fourth.png)

Lastly, if you go to the artifacts tab, you van find the different artifacts that you saved. This includes your model and the figures that you saved. As you can see, the model is located in a seperate folder. As you can see in this screenshot, MLflow gives an explenation how to use your mlflow model for inference. This means that this model can easily be sent to a different person and tested by them without a lot of coding.

![Mlflow_artifacts](images/mlflow_fifth.png)

## mlruns and mlartifacts folders

When you start running your experiment you will see that MLflow has created two folder in your directory. These are the mlruns and the mlartifacts folders. There MLflow will keep track of the different runs and artifacts locally. This can be changed if you want to set this up on a specific server. 
