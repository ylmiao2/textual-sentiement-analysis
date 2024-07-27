This is a repo for textual sentiment analysis. Two models are implemented: an LSTM and a BERT model.

## Data
A toydata is provided under the `./data` folder, which contains 100 samples with 3 kinds of sentiments: positive, negative and neutral. The sentiment labels are encoded as integers: 0 for negative, 1 for neutral, and 2 for positive.


## Usage
To train the models, first need to write your `config.yaml` file. Then, run the following command:
```
python run.py --config config.yaml
```
The trained models will be saved under the `./ckpt` folder.