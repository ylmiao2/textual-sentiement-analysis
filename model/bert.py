import torch
import torch.nn as nn
import transformers


class BERTClassifier(torch.nn.Module):
    def __init__(self, num_class):
        super(BERTClassifier, self).__init__()
        # load the pretrained Bert model
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')

        # freeze the parameters in Bert model
        # for p in self.bert.parameters():
        #     p.requires_grad = False

        # design the fully connected layers
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(768, 256),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.2),

        #     torch.nn.Linear(256, num_class)
        # )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            torch.nn.Linear(768, num_class)
        )
    
    def forward(self, ids, masks):
        _, pooler_output = self.bert(input_ids=ids, attention_mask=masks, return_dict=False)
        output = self.fc_layers(pooler_output)
        return output