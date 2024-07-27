from tqdm import tqdm
import torch

def train_bert(args, model, dataloader, criterion, optimizer):

    # put the model in train mode
    model.train()
    epoch_loss = []
    epoch_acc = []
    count_samples = 0

    for batch in tqdm(dataloader):
        ids = batch['ids'].to(args.model['device'], dtype = torch.long)
        masks = batch['masks'].to(args.model['device'], dtype = torch.long)
        labels = batch['labels'].to(args.model['device'], dtype = torch.long)

        # put into model
        output = model(ids, masks) # [batchsize, num_class]

        # compute the loss
        loss = criterion(output, labels)

        # get the predicted label
        pred_label = torch.argmax(torch.softmax(output, dim=1), dim=1)

        # calculate the accuracy
        acc = ((pred_label.view(-1) == labels.view(-1)).sum()).item()
        epoch_acc.append(acc)
        count_samples += len(labels.view(-1))

        # update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # add the loss to epoch loss
        epoch_loss.append(loss.item())
    
    # compute the average acc and loss
    epoch_train_loss = sum(epoch_loss)/len(epoch_loss)
    epoch_train_acc = sum(epoch_acc)/count_samples

    return epoch_train_loss, epoch_train_acc