from tqdm import tqdm
from torcheval.metrics import BinaryAccuracy

def evaluate_model(model, test_dataset, device):
    """Train the given model on the train dataset and evaluate on the val dataset"""

    criterion = model.criterion

    # set comparison values
    history = dict(test_loss=[], test_acc=[])
    
    # train the model. Run the model on the inputs, calculate the losses, do backpropagation

    for inputs, labels in tqdm(test_dataset):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # prepare data and evaluate model
        predictions = model(inputs)
        
        # calculate loss and accuracy
        loss = criterion(predictions.flatten(), labels)
        accuracy_metric = BinaryAccuracy(threshold=0.7)
        accuracy_metric.update(predictions.flatten(), labels)
        accuracy = accuracy_metric.compute()

        # save loss and accuracy values
        history['test_loss'].append(loss.item())
        history['test_acc'].append(accuracy.item())

    return history