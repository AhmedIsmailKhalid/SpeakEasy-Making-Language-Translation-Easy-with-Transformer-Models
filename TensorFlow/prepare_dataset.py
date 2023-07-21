import numpy as np
from pickle import dump, HIGHEST_PROTOCOL
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64

class PrepareDataset:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_tokenizer(self, dataset):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataset)

        return tokenizer

    def get_seq_length(self, dataset):
        return max(len(seq.split()) for seq in dataset)

    def get_vocab_size(self, tokenizer, dataset):
        tokenizer.fit_on_texts(dataset)

        return len(tokenizer.word_index) + 1

    def encoding_pad(self, dataset, tokenizer, seq_length):
        x = tokenizer.texts_to_sequences(dataset)
        x = pad_sequences(x, maxlen=seq_length, padding='post')
        x = convert_to_tensor(x, dtype=int64)

        return x

    '''def save_tokenizer(self, tokenizer, name):
        tokenizer_json = tokenizer.to_json()
        with open(name + '_tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(dumps(tokenizer_json, ensure_ascii=False))'''
    def save_tokenizer(self, tokenizer, name):
        with open(name + '_tokenizer.pkl', 'wb') as handle:
            dump(tokenizer, handle, protocol=HIGHEST_PROTOCOL)


    def load_data(self, filename):
        with open(filename, 'r', encoding='utf8') as file:
            return file.read().split('\n')

    def __call__(self):
        train_en = self.load_data('data/train.en')[:100]
        train_de = self.load_data('data/train.de')[:100]

        val_en = self.load_data('data/val.en')[:100]
        val_de = self.load_data('data/val.de')[:100]

        test_en = self.load_data('data/test_2018_flickr.en')[:50]
        test_de = self.load_data('data/test_2018_flickr.de')[:50]

        # Include start and end of string tokens
        train_en = ["<START> " + txt + " <EOS>" for txt in train_en]
        train_de = ["<START> " + txt + " <EOS>" for txt in train_de]

        val_en = ["<START> " + txt + " <EOS>" for txt in val_en]
        val_de = ["<START> " + txt + " <EOS>" for txt in val_de]

        test_en = ["<START> " + txt + " <EOS>" for txt in test_en]
        test_de = ["<START> " + txt + " <EOS>" for txt in test_de]

        # Prepare tokenizer for the encoder input
        enc_tokenizer = self.create_tokenizer(train_en)
        enc_seq_length = self.get_seq_length(train_en)
        enc_vocab_size = self.get_vocab_size(enc_tokenizer, train_en)

        # Prepare tokenizer for the decoder input
        dec_tokenizer = self.create_tokenizer(train_de)
        dec_seq_length = self.get_seq_length(train_de)
        dec_vocab_size = self.get_vocab_size(dec_tokenizer, train_de)

        # Encode and pad the training input
        X_train = self.encoding_pad(train_en, enc_tokenizer, enc_seq_length)
        y_train = self.encoding_pad(train_de, dec_tokenizer, dec_seq_length)

        # Encode and pad the validation input
        X_valid = self.encoding_pad(val_en, enc_tokenizer, enc_seq_length)
        y_valid = self.encoding_pad(val_de, dec_tokenizer, dec_seq_length)

        # Encode and pad the test input
        X_test = self.encoding_pad(test_en, enc_tokenizer, enc_seq_length)
        y_test = self.encoding_pad(test_de, dec_tokenizer, dec_seq_length)

        # Save the encoder tokenizer
        self.save_tokenizer(enc_tokenizer, 'enc')

        # Save the decoder tokenizer
        self.save_tokenizer(dec_tokenizer, 'dec')


        return (X_train, y_train, X_valid, y_valid, X_test, y_test, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size)
