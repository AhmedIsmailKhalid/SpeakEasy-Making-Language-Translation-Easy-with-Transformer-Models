import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from pickle import dump, HIGHEST_PROTOCOL

class PrepareDataset:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = get_tokenizer('basic_english')

    def tokenize(self, data_iter):
        data_iter = (self.tokenizer(txt) for txt in data_iter)
        vocab = build_vocab_from_iterator(data_iter, specials=['<UNK>', '<PAD>', '<START>', '<EOS>'])
        vocab.set_default_index(vocab['<UNK>'])

        return vocab

    def data_process(self, data_iter, vocab, max_len):
        data = []
        for item in data_iter:
            tokens = self.tokenizer(item)
            if len(tokens) > max_len:  # if sentence is longer than max_len, truncate it
                tokens = tokens[:max_len]
            else:  # if sentence is shorter than max_len, pad it
                tokens += ['<pad>'] * (max_len - len(tokens))
            data.append(torch.tensor([vocab[token] for token in tokens], dtype=torch.long))
        return torch.stack(data)


    def save_tokenizer(self, tokenizer, name):
        with open(name + '_tokenizer.pkl', 'wb') as handle:
            dump(tokenizer, handle, protocol=HIGHEST_PROTOCOL)

    def load_data(self, filename):
        with open(filename, 'r', encoding='utf8') as file:
            return file.read().split('\n')

    def __call__(self):
        train_en = self.load_data('data/train.en')[:5000]
        train_de = self.load_data('data/train.de')[:5000]

        val_en = self.load_data('data/val.en')[:100]
        val_de = self.load_data('data/val.de')[:100]

        test_en = self.load_data('data/test_2018_flickr.en')[:100]
        test_de = self.load_data('data/test_2018_flickr.de')[:100]

        train_en = ["<START> " + txt + " <EOS>" for txt in train_en]
        train_de = ["<START> " + txt + " <EOS>" for txt in train_de]

        val_en = ["<START> " + txt + " <EOS>" for txt in val_en]
        val_de = ["<START> " + txt + " <EOS>" for txt in val_de]

        test_en = ["<START> " + txt + " <EOS>" for txt in test_en]
        test_de = ["<START> " + txt + " <EOS>" for txt in test_de]

        enc_vocab = self.tokenize(train_en)
        dec_vocab = self.tokenize(train_de)

        # Get the encoder and decoder seq_lengths
        enc_seq_length = max(len(self.tokenizer(txt)) for txt in train_en) #self.get_seq_length(train_en)
        dec_seq_length = max(len(self.tokenizer(txt)) for txt in train_de) #self.get_seq_length(train_de)

        # Get the encoder and decoder vocab_sizes
        enc_vocab_size = len(enc_vocab)
        dec_vocab_size = len(dec_vocab)

        X_train = self.data_process(train_en, enc_vocab, enc_seq_length)
        y_train = self.data_process(train_de, dec_vocab, dec_seq_length)

        X_valid = self.data_process(val_en, enc_vocab, enc_seq_length)
        y_valid = self.data_process(val_de, dec_vocab, dec_seq_length)

        X_test = self.data_process(test_en, enc_vocab, enc_seq_length)
        y_test = self.data_process(test_de, dec_vocab, dec_seq_length)


        self.save_tokenizer(enc_vocab, 'enc')
        self.save_tokenizer(dec_vocab, 'dec')

        return (X_train, y_train, X_valid, y_valid, X_test, y_test, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size)

