import torch
import optuna
from datetime import datetime
import os
import mlflow
import numpy as np
import torch.nn as nn 
import optuna
from torcheval.metrics import BinaryAccuracy

def save_best_model(best_acc, best_loss, best_weights, trial:optuna.Trial):
        """Save the weights of the best model, together with the encoder"""

        save_dir = "model_params/" 
        print("saving best model epoch, acc {:.4f} and loss {:.4f}".format(best_acc, best_loss))
        torch.save(best_weights, os.path.join(save_dir, '{}-{:.4f}-full_model.pth'.format(trial._trial_id, best_loss)))
                   

def train_model(model, train_dataset, val_dataset, n_epochs, device, learning_rate, trial):
    """Train the given model on the train dataset and evaluate on the val dataset"""

    # set optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = model.criterion

    # set comparison values
    history = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
    best_loss = np.inf
    best_weights = None
    best_acc = None
    
    # train the model. Run the model on the inputs, calculate the losses, do backpropagation
    for epoch in range(n_epochs):

        # training set
        model = model.train()

        train_losses = []
        train_accs = []

        for inputs, labels in train_dataset:
            optimizer.zero_grad()
            # prepare data and evaluate model
            labels = labels.to(device)
            predictions = model(inputs)
            predictions = predictions.to(device)
            
            # calculate loss and accuracy
            loss = criterion(predictions.flatten(), labels)
            accuracy_metric = BinaryAccuracy(threshold=0.5)
            accuracy_metric.update(predictions.flatten(), labels)
            accuracy = accuracy_metric.compute()

            # backwards propegation
            loss.backward()
            optimizer.step()

            # save loss and accuracy values
            train_losses.append(loss.item())
            train_accs.append(accuracy.item())  

        # validation set 
        model = model.eval()
        val_losses = []
        val_accs = []
        # run the model and loss on the validation function
        with torch.no_grad():
            for inputs, labels in val_dataset:
                # prepare data and evaluate model
                labels = labels.to(device)
                inputs = inputs.to(device)
                predictions = model(inputs)
                predictions = predictions.to(device)

                # calculate loss and accuracy
                loss = criterion(predictions.flatten(), labels)
                accuracy_metric = BinaryAccuracy(threshold = 0.7)
                accuracy_metric.update(predictions.flatten(), labels)
                accuracy = accuracy_metric.compute()

                # save loss and accuracy values
                val_losses.append(loss.item())
                val_accs.append(accuracy.item())
        
        # get the losses and accs from the epoch
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = np.mean(train_accs)
        val_acc = np.mean(val_accs)

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)
        mlflow.log_metrics({
            "train_acc": train_acc,
            "val_acc": val_acc
        }, step=epoch)


        # decide if this version of the model is the best
        loss = float(val_loss)
        if loss < best_loss:
            best_acc = accuracy
            best_loss = loss
            best_weights = model.state_dict()

        # optuna prune trial
        if trial != None: 
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # save loss and accuracy functions
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        text = f'Epoch = {epoch}, train loss = {train_loss}, val loss = {val_loss}'
        print(text)


    # save the best model
    save_best_model(best_acc, best_loss, best_weights, trial)
    return model, history