import torch
from tqdm import tqdm

def train_lstm(args, dataloader, model, criterion, optimizer):

    model.train()
    epoch_losses = []
    epoch_accs = []

    for batch in tqdm(dataloader):
        ids = batch['ids'].to(args.model['device'])
        length = batch['length']
        label = batch['labels'].to(args.model['device'])
        prediction = model(ids, length)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())

    return epoch_losses, epoch_accs

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy