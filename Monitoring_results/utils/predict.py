from tqdm import tqdm
from torcheval.metrics import BinaryAccuracy

def evaluate_model(model, test_dataset):
    """Train the given model on the train dataset and evaluate on the val dataset"""

    criterion = model.criterion
    # set comparison values
    predicts = []
    # train the model. Run the model on the inputs, calculate the losses, do backpropagation

    for inputs, labels in tqdm(test_dataset):
        # prepare data and evaluate model
        predictions = model(inputs)
        predicts.append(predictions.flatten())
    
    return predicts