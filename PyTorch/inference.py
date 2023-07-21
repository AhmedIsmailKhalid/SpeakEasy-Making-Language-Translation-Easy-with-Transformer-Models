import pickle
import random
import numpy as np
import torch
from model import Transformer
from prepare_dataset import PrepareDataset
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab


class Translate:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = get_tokenizer('basic_english')
        self.load_params()
        self.model = Transformer(self.num_layers, self.model_dim, self.num_heads, self.feedforward_dim, 
                                 self.input_vocab_size, self.target_vocab_size,
                                 self.pe_input, self.pe_target, dropout_rate=self.dropout_rate).to(self.device)
        self.model.load_state_dict(torch.load('./saved_model/saved_model.pth', map_location=self.device))
        self.input_stoi, self.input_itos, self.target_stoi, self.target_itos = self.load_vocabs()

        _, _, _, _, self.X_test, self.y_test, _, _, _, _ = PrepareDataset()()

    def load_params(self):
        with open('dataset_params.pkl', 'rb') as f:
            params = pickle.load(f)
            self.pe_input = params['enc_seq_length']
            self.pe_target = params['dec_seq_length']
            self.input_vocab_size = params['enc_vocab_size']
            self.target_vocab_size = params['dec_vocab_size']

        with open('model_params.pkl', 'rb') as f:
            params = pickle.load(f)
            self.num_layers = params['num_layers']
            self.model_dim = params['model_dim']
            self.num_heads = params['num_heads']
            self.feedforward_dim = params['feedforward_dim']
            self.dropout_rate = params['dropout_rate']

    def load_vocabs(self):
        with open('enc_tokenizer.pkl', 'rb') as f:
            input_vocab = pickle.load(f)

        with open('dec_tokenizer.pkl', 'rb') as f:
            target_vocab = pickle.load(f)

        return input_vocab.get_stoi(), input_vocab.get_itos(), target_vocab.get_stoi(), target_vocab.get_itos()

    def translate(self, sentence):
        self.model.eval()
        with torch.no_grad():
            src = self.tokenizer(sentence)
            src = torch.LongTensor([[self.input_stoi.get(token, self.input_stoi['<UNK>']) for token in src]])
            src = src.to(self.device)

            tgt = torch.ones((1, 1)).fill_(self.target_stoi["<START>"]).type_as(src)
            for i in range(self.pe_target):
                output = self.model(src, tgt)
                output = output.squeeze(0).detach().cpu().numpy()
                output = np.argmax(output, axis=-1)
                tgt = torch.cat((tgt, torch.ones((1, 1)).type_as(src).fill_(output[-1])), dim=1)
                if output[-1] == self.target_stoi["<EOS>"]:
                    break

            return " ".join([self.target_itos[tok] for tok in tgt.tolist()[0] if self.target_itos[tok] not in ['<START>', '<start>', '<EOS>', '<eos>',
                                                                                                                      '<PAD>', '<pad>','<UNK>', '<unk>']])


    def translate_test_dataset(self, option=None):
        if self.X_test is None or self.y_test is None:
            print("Test data not loaded.")
            return

        if option == 1:
            idx = random.randint(0, len(self.X_test)-1)
            original_sentence = ' '.join([self.input_itos[i] for i in self.X_test[idx] if self.input_itos[i] not in ['<START>', '<start>', '<EOS>', '<eos>',
                                                                                                                     '<PAD>', '<pad>','<UNK>', '<unk>']])
            print('Original sentence: ', original_sentence)
            print('Translation: ', self.translate(original_sentence), '\n')
            #print()
        elif option == 2:
            n = int(input("Enter the number of sentences to translate: "))
            for _ in range(n):
                idx = random.randint(0, len(self.X_test)-1)
                original_sentence = ' '.join([self.input_itos[i] for i in self.X_test[idx] if self.input_itos[i] not in ['<START>', '<start>', '<EOS>', '<eos>',
                                                                                                                     '<PAD>', '<pad>','<UNK>', '<unk>']])
                print('Original sentence: ', original_sentence)
                print('Translation: ', self.translate(original_sentence), '\n')
                #print()
        elif option == 3:
            for sentence in self.X_test:
                original_sentence = ' '.join([self.input_itos[i] for i in sentence if self.input_itos[i] not in ['<START>', '<start>', '<EOS>', '<eos>',
                                                                                                                     '<PAD>', '<pad>','<UNK>', '<unk>']])
                print('Original sentence: ', original_sentence)
                print('Translation: ', self.translate(original_sentence), '\n')
                #print()


    def start(self):
        while True:
            print("1. Translate a custom sentence")
            print("2. Translate test dataset")
            print("3. Exit")
            choice = int(input("Please enter your choice: "))

            if choice == 1:
                sentence = input("Please enter the sentence to translate: ")
                translation = self.translate(sentence)
                print('Translation: ', translation, '\n')
            elif choice == 2:
                print("\t1. Translate a random sentence")
                print("\t2. Translate n random sentences")
                print("\t3. Translate the entire test dataset")
                print("\t4. Go back to the main menu")
                option = int(input("Please enter your choice: "))
                if option in [1, 2, 3]:
                    self.translate_test_dataset(option)
                elif option == 4:
                    continue
                else:
                    print("Invalid choice. Please enter a valid choice.\n")
            elif choice == 3:
                break
            else:
                print("Invalid choice. Please enter a valid choice.\n")


if __name__ == "__main__":
    translate = Translate()
    translate.start()
