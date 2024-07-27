import datasets
import torchtext
from transformers import BertTokenizer


class DataProcessor():
    def __init__(self, max_len, model):
        self.max_len = max_len

        self.model = model
        assert model in ['LSTM', "bert-base-uncased"], f"{self.model} not supported"
    

    def split_dataset(self, dataset):
        splited_dataset = dataset.train_test_split(test_size = 0.1, stratify_by_column="labels")
        train_ = splited_dataset['train']
        splited_ = train_.train_test_split(test_size = 0.1, stratify_by_column="labels")
        train_set, valid_set = splited_["train"], splited_["test"]
        test_set = splited_dataset['test']
        return train_set, valid_set, test_set
    

    def process_for_LSTM(self, dataset):
        def tokenize_data(example):
            tokens = tokenizer(example['text'])[:self.max_len]
            length = len(tokens)
            example['tokens'] = tokens
            example['length'] = length
            return example

        def yield_tokens(data_iter):
            for item in data_iter:
                yield item['tokens']
        
        def vectorize_data(example):
            ids = [vocab[token] for token in example['tokens']] # token is a list
            example['ids'] = ids
            return example

        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        dataset = dataset.map(tokenize_data)
        vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(dataset), min_freq=5, specials=['<unk>', '<pad>'])
        vocab.set_default_index(vocab['<unk>'])
        pad_index = vocab['<pad>'] # pad token
        dataset = dataset.map(vectorize_data)
        
        train_set, valid_set, test_set = self.split_dataset(dataset)

        return len(vocab), pad_index, train_set, valid_set, test_set

    
    def process_for_BERT(self, dataset):
        def preprocess(example):
            # tokenize the texts
            tokens = tokenizer(example["text"], padding='max_length', max_length = self.max_len,
                                    truncation=True, return_tensors="pt")
            example["ids"] = tokens["input_ids"].squeeze()
            example["masks"] = tokens["attention_mask"].squeeze()
            return example
        
        tokenizer = BertTokenizer.from_pretrained(self.model)
        dataset = dataset.map(preprocess, remove_columns = ["text"])

        train_set, valid_set, test_set = self.split_dataset(dataset)

        return train_set, valid_set, test_set


    def process(self, input_data):
        dataset = datasets.Dataset.from_pandas(input_data)
        dataset = dataset.class_encode_column("labels")

        if self.model == 'LSTM':
            return self.process_for_LSTM(dataset)
        elif self.model == "bert-base-uncased":
            return self.process_for_BERT(dataset)
        
