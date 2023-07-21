import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pickle import load
from model import TransformerModel
from prepare_dataset import PrepareDataset
from random import choice, sample
import numpy as np

class Translate:
    X_test = None
    y_test = None

    def __init__(self):
        self.load_params()
        self.model = TransformerModel(self.enc_vocab_size, self.dec_vocab_size, self.enc_seq_length, self.dec_seq_length,
                                      self.num_heads, self.keys_dim, self.values_dim, self.model_dim, self.feedforward_dim, self.num_layers,
                                      self.dropout_rate)
        self.model.load_weights('weights/wghts2.ckpt')  # load the saved weights
        self.load_tokenizers()

    def load_params(self):
        with open('dataset_params.pkl', 'rb') as f:
            self.enc_seq_length, self.dec_seq_length, self.enc_vocab_size, self.dec_vocab_size = load(f)

        with open('model_params.pkl', 'rb') as f:
            model_params = load(f)
            
        self.num_heads = model_params['num_heads']
        self.keys_dim = model_params['keys_dim']
        self.values_dim = model_params['values_dim']
        self.model_dim = model_params['model_dim']
        self.feedforward_dim = model_params['feedforward_dim']
        self.num_layers = model_params['num_layers']
        self.dropout_rate = model_params['dropout_rate']

    def load_tokenizers(self):
        with open('enc_tokenizer.pkl', 'rb') as f:
            self.enc_tokenizer = load(f)

        with open('dec_tokenizer.pkl', 'rb') as f:
            self.dec_tokenizer = load(f)

    def prepare_sentence(self, sentence):
        # if sentence is already a sequence, just pad it
        if isinstance(sentence, np.ndarray):
            sentence = pad_sequences([sentence], maxlen=self.enc_seq_length, padding='post')
        else:  # else convert text to sequence
            sentence = self.enc_tokenizer.texts_to_sequences([sentence])
            sentence = pad_sequences(sentence, maxlen=self.enc_seq_length, padding='post')
        
        sentence = tf.convert_to_tensor(sentence)
        return sentence

    def prepare_output(self):
        start_token = self.dec_tokenizer.texts_to_sequences(['<START>'])
        end_token = self.dec_tokenizer.texts_to_sequences(['<EOS>'])

        return tf.convert_to_tensor(start_token), tf.convert_to_tensor(end_token)

    def translate(self, sentence):
        encoder_input = self.prepare_sentence(sentence)
        output_start, output_end = self.prepare_output()
        
        output = output_start
        for i in range(self.dec_seq_length):
            prediction = self.model(encoder_input, output, training=False)

            predicted_id = tf.argmax(prediction[:, -1:, :], axis=-1)
            output = tf.concat([output, tf.cast(predicted_id, tf.int32)], axis=-1)

            if tf.cast(predicted_id, tf.int32) == tf.cast(output_end, tf.int32):
                break

        translated_sentence = self.dec_tokenizer.sequences_to_texts(output.numpy())[0]
        return translated_sentence#' '.join(translated_sentence.split(' '))[1:-1]

    def prepare_test_data(self):
        if Translate.X_test is None and Translate.y_test is None:
            prepare = PrepareDataset()
            _, _, _, _, Translate.X_test, Translate.y_test, _, _, _, _ = prepare()

    def translate_test_dataset(self, n=None, random=False):
        self.prepare_test_data()

        if n is not None:
            indices = np.random.randint(0, len(Translate.X_test), n) if random else range(n)

            for i in indices:
                #print(f"Sentence: {self.enc_tokenizer.sequences_to_texts([Translate.X_test[i].numpy()])[0]}")
                source_sentence = self.enc_tokenizer.sequences_to_texts([Translate.X_test[i].numpy()])[0]
                target_sentence = self.translate(Translate.X_test[i].numpy())
                
                print('Sentence    :', ' '.join(source_sentence.split(' ')[1:-1]))
                print('Translation :', ' '.join(target_sentence.split(' ')[1:-1]), '\n')

                #print(f"Translation: {self.translate(Translate.X_test[i].numpy())}\n")
        else:
            for i in range(len(Translate.X_test)):
                #print(f"Sentence: {self.enc_tokenizer.sequences_to_texts([Translate.X_test[i].numpy()])[0]}")
                #print(f"Translation: {self.translate(Translate.X_test[i].numpy())}\n")
                print()
                print('Sentence    :', ' '.join(source_sentence.split(' ')[1:-1]))
                print('Translation :', ' '.join(target_sentence.split(' ')[1:-1]), '\n')

    def translate_custom_sentence(self):
        sentence = input("Please enter the sentence to translate: ")
        translation = self.translate(sentence)
        print('Translation: ', ' '.join(translation.split(' ')[1:-1]), '\n')

    def start(self):
        while True:
            print("1. Translate a custom sentence")
            print("2. Translate test dataset")
            print("3. Exit")
            choice = int(input("Please enter your choice: "))

            if choice == 1:
                self.translate_custom_sentence()
            elif choice == 2:
                print("\t1. Translate random sentence")
                print("\t2. Translate entire test dataset")
                print("\t3. Translate n random sentences")
                print("\t4. Go back to main menu")
                sub_choice = int(input("Please enter your choice: "))

                if sub_choice == 1:
                    print()
                    self.translate_test_dataset(n=1, random=True)
                elif sub_choice == 2:
                    self.translate_test_dataset()
                elif sub_choice == 3:
                    n = int(input("Please enter the number of sentences to translate: "))
                    print()
                    self.translate_test_dataset(n=n, random=True)
            elif choice == 3:
                break
            else:
                print("Invalid choice. Please enter a valid choice.\n")


if __name__ == "__main__":
    translate = Translate()
    translate.start()
