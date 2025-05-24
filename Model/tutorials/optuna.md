# Optuna implementation

Optuna is used to automatically optimize the hyperparameters of your machine learning model. To start implementing optuna we work in our "main.py" file.

## imports

```
    import optuna
```

## Start a study 

```
study_name = "Example"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize", 
            sampler = optuna.samplers.TPESampler(),
            pruner = optuna.pruners.MedianPruner(n_warmup_steps = 5, n_startup_trials = 5)
            )
```

To create a study, you can use the optuna.create_study function. It has the following parameters:
- study_name: The name of the study.
- direction: This can be "minimize" or "maximize" depending on the on what you want to optimize.
- sampler: The type of search used for the hyperparameter tuning. More information about the different samplers can be found [here](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html)
- pruner: The type of pruner used to prune trials when they do not preform well. More information about the different samplers can be found [here](https://optuna.readthedocs.io/en/stable/reference/pruners.html)

Once you created your study, you can start the optimization process with the following command

```
study.optimize(objective_function, n_trials=10 )
```

- Objective_function: the name of the function that needs to be optimized
- n_trials: The number of trials that are done
There are some optional parameters you can set here. For example the n_jobs parameter can be set to a value larger than 1 to run parallel jobs. To learn more about these parameters follow this [link](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize)

## The objective function

Optuna has a type of variable called a trial. A trial contains all information about this loop of the objective function. To be able to access the functionality and parameters of optuna for your trial this has to be given to the objective function.

```
def objective_function(trial:optuna.trial):
    // objective functionality
```

To be able to implement a search the possible choises of hyperparameters has to be given to optune. This can be done with the several "suggest" functions in your objective function. At this point in the code optuna will choose a value for the given variable depending on the choise options. Here are two examples. To find out more about suggest functions follow this [link](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial)

```
integer = trial.suggest_int("integer_name", min_value, max_value)  
float_value = trial.suggest_float("float_name", min_value, max_value)
```
lastly we need to return a value in our objective function. This is the metric which we want to optimize. In our example this is the lowest validation loss of the training loop. This corresponds with the best saved model.

```
def objective_function(trial:optuna.trial):
    // functionality
    return best_validation_loss
```

## pruning

If you have implemented the above code, you can already run a hyperparameter search. However, at this point no pruning will be done, which can make the search take a long time. The implementation of the pruning has to be done in the training loop. This means we need to give the trial variable as a variable to our training loop. When this is done, we save our validation loss during the training loop using this trial variable. Then we let optuna decide if this trial should be pruned or not. This is done using the following code.

```
    if trial != None: 
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
```

## Analysing your study

When the study is done, we can access certain variables and create certain plots to analyze the results of the study. This is done after our study.optimize call.

### looking at parameters
First, to get the parameters of the best performing trial of the study, we can implement the following code.

```
best_trial = study.best_trial
print("best trial: " + str(best_trial.value))
for key, value in best_trial.params.items():
        print("     {}: {}".format(key, value))
```


### creating plots
To visualize some interesting plots we can use the visualization of the optuna library. To find an overview of the different available plots you can follow this [link](https://optuna.readthedocs.io/en/stable/reference/visualization/index.html). Here you can find two examples.

```
fig1 = optuna.visualization.plot_optimization_history(study)
fig2 = optuna.visualization.plot_param_importances(study)
```
These are plotly figures which show the results of the objective function over time and the importance of the different parameters used in the study. The plots are created using plotly which can be saved as an html file.

```
file_dir = "example_directory/graphs"
fig1.write_html(file_dir + "_optimization_history.html")
fig2.write_html(file_dir + "_params_importance.html")
```

If you would want to save the plots as a png, you can use matplotlib to create the figures in the followin way.

```
fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
```
Do note that these are still experimental functions and can contain some errors. For more information you can look [here](https://optuna.readthedocs.io/en/stable/reference/visualization/matplotlib/index.html)



