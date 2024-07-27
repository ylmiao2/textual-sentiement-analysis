import argparse
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from transformers import default_data_collator

from tqdm.auto import tqdm
from data.data_processing import DataProcessor
from model import lstm, bert
from utils.load_yaml import load_yaml
from utils.utils import collate_fn
from train.train_bert import train_bert
from train.train_lstm import train_lstm
from eval.eval_bert import eval_bert
from eval.eval_lstm import eval_lstm



def run(args):
    # initiate the data processor
    dataprocesser = DataProcessor(args.model['max_length'], args.model['name'])

    # get data, initiate the model
    if args.model['name'] == 'LSTM':
        vocab_size, pad_index, train_set, valid_set, test_set = dataprocesser.process(pd.read_csv(args.input_path))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train['batch_size'], collate_fn=lambda x: collate_fn(x, pad_index), shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.train['batch_size'], collate_fn=lambda x: collate_fn(x, pad_index), shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.train['batch_size'], collate_fn=lambda x: collate_fn(x, pad_index), shuffle=False)

        model = lstm.LSTM(
            vocab_size=vocab_size, 
            pad_index=pad_index,
            embedding_dim=args.model['embedding_dim'], 
            hidden_dim=args.model['hidden_dim'], 
            num_class=args.model['num_class'], 
            num_layers=args.model['num_layers'], 
            bidirectional=args.model['bidirectional'], 
            dropout=args.model['dropout']
        ).to(args.model['device'])

        # define the loss function
        optimizer = optim.Adam(model.parameters(), lr=args.train['learning_rate'])
        
        # define the loss function
        criterion = torch.nn.CrossEntropyLoss()

        best_valid_loss = float('inf')

        for epoch in tqdm(range(args.train['num_epochs'])):
            train_loss, train_acc = train_lstm(args, train_loader, model, criterion, optimizer)
            valid_loss, valid_acc = eval_lstm(args, valid_loader, model, criterion)
            
            epoch_train_loss = np.mean(train_loss)
            epoch_train_acc = np.mean(train_acc)
            epoch_valid_loss = np.mean(valid_loss)
            epoch_valid_acc = np.mean(valid_acc)
            
            if epoch_valid_loss < best_valid_loss:
                best_valid_loss = epoch_valid_loss
                torch.save(model.state_dict(), './ckpt/lstm.pth')
            
            print(f"[ Train | {epoch+1}/{args.train['num_epochs']} ] loss:{epoch_train_loss:6f}, acc:{epoch_train_acc:6f}")
            print(f"[ Valid | {epoch+1}/{args.train['num_epochs']} ] loss:{epoch_valid_loss:6f}, acc:{epoch_valid_acc:6f}")

            model.load_state_dict(torch.load('./ckpt/lstm.pth'))
            test_loss, test_acc = eval_lstm(args, test_loader, model, criterion)

            epoch_test_loss = np.mean(test_loss)
            epoch_test_acc = np.mean(test_acc)

            print(f"[ Test ] test_loss: {epoch_test_loss:.6f}, test_acc: {epoch_test_acc:.6f}")
                
    elif args.model['name'] == "bert-base-uncased":
        train_set, valid_set, test_set = dataprocesser.process(pd.read_csv(args.input_path))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train['batch_size'], collate_fn=default_data_collator, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.train['batch_size'], collate_fn=default_data_collator, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.train['batch_size'], collate_fn=default_data_collator, shuffle=False)

        model = bert.BERTClassifier(num_class= args.model['num_class']).to(args.model['device'])
    
        # define the loss function
        optimizer = optim.Adam(model.parameters(), lr=args.train['learning_rate'])
        
        # define the loss function
        criterion = torch.nn.CrossEntropyLoss()

        best_valid_loss = float('inf')

        for epoch in range(args.train['num_epochs']):
            # train the model
            train_loss, train_acc = train_bert(args, model, train_loader, criterion, optimizer)

            # save the model
            if train_loss < best_valid_loss:
                best_valid_loss = train_loss
                torch.save(model.state_dict(), f"./ckpt/bert.pth")

            # evaluation the model
            eval_loss, eval_acc = eval_bert(args, model, valid_loader, criterion)

            # print the result
            print(f"[ Train | {epoch+1}/{args.train['num_epochs']} ] loss:{train_loss:6f}, acc:{train_acc:6f}")
            print(f"[ Valid | {epoch+1}/{args.train['num_epochs']} ] loss:{eval_loss:6f}, acc:{eval_acc/len(valid_set):6f}")

            model.load_state_dict(torch.load('./ckpt/lstm.pth'))
            test_loss, test_acc = eval_bert(args, model, test_loader, criterion)

            print(f"[ Test ]test_loss: {test_loss:.6f}, test_acc: {test_acc/len(test_set):.6f}")

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml', 
        help='path to the config')
    
    args = argparser.parse_args()

    args = load_yaml(args)

    run(args)