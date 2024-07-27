import torch
import torch.nn as nn

def collate_fn(batch, pad_index):
    batch_ids = [torch.tensor(i['ids']) for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_length = [torch.tensor(i['length']) for i in batch]
    batch_length = torch.stack(batch_length)
    batch_label = [torch.tensor(i['labels']) for i in batch]
    batch_label = torch.stack(batch_label)
    batch = {'ids': batch_ids,
             'length': batch_length,
             'labels': batch_label}
    return batch