import torch
from tqdm import tqdm

def eval_bert(args, model, dataloader, criterion):

    # put the model in train mode
    model.eval()

    epoch_loss = []
    epoch_acc = []

    with torch.no_grad():
      for batch in tqdm(dataloader):
        ids = batch['ids'].to(args.model['device'], dtype = torch.long)
        masks = batch['masks'].to(args.model['device'], dtype = torch.long)
        labels = batch['labels'].to(args.model['device'], dtype = torch.long)

        # put into model
        output = model(ids, masks)

        # compute the loss
        loss = criterion(output, labels)

        # get the predicted label
        pred_label = torch.argmax(torch.softmax(output, dim=1), dim=1)

        # calculate the accuracy
        acc = ((pred_label.view(-1) == labels.view(-1)).sum()).item()
        epoch_acc.append(acc)
        
        # add the loss to epoch loss
        epoch_loss.append(loss.item())
    
    # compute the average acc and loss
    epoch_eval_loss = sum(epoch_loss)/len(dataloader)
    epoch_eval_acc = sum(epoch_acc)

    return epoch_eval_loss, epoch_eval_acc
