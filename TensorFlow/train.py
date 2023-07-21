import warnings
import tensorflow as tf
from time import time
from pickle import dump, load
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import data, math, reduce_sum, cast, equal, argmax, float32, GradientTape
from tensorflow.keras.losses import sparse_categorical_crossentropy
from model import TransformerModel
from prepare_dataset import PrepareDataset
warnings.filterwarnings('ignore')

class TrainModel:
    def __init__(self, num_heads=8, keys_dim=64, values_dim=64, model_dim=512, feedforward_dim=2048, num_layers=6, epochs=2,
                 batch_size=64, beta_1=0.9, beta_2=0.98, epsilon=1e-9, dropout_rate=0.1):
        self.num_heads = num_heads
        self.keys_dim = keys_dim
        self.values_dim = values_dim
        self.model_dim = model_dim
        self.feedforward_dim = feedforward_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.dropout_rate = dropout_rate

        self.train_loss = Mean(name='train_loss')
        self.train_accuracy = Mean(name='train_accuracy')
        self.val_loss = Mean(name='val_loss')
        self.val_accuracy = Mean(name='val_accuracy')

    class CustomLRSchedule(LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000, **kwargs):
            super().__init__(**kwargs)
            self.d_model = cast(d_model, float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step_num):
            arg1 = step_num ** -0.5
            arg2 = step_num * (self.warmup_steps ** -1.5)

            return (self.d_model ** -0.5) * math.minimum(arg1, arg2)

    def calculate_loss(self, target, prediction):
        mask = math.logical_not(equal(target, 0))
        mask = cast(mask, float32)

        loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * mask

        return reduce_sum(loss) / reduce_sum(mask)

    def calculate_accuracy(self, target, prediction):
        mask = math.logical_not(equal(target, 0))
        accuracy = equal(target, argmax(prediction, axis=2))
        accuracy = math.logical_and(mask, accuracy)

        mask = cast(mask, float32)
        accuracy = cast(accuracy, float32)

        return reduce_sum(accuracy) / reduce_sum(mask)

    def calculate_bleu(self, references, predictions):
        return corpus_bleu(references, predictions)

    def calculate_rouge(self, references, predictions):
        rouge = Rouge()
        scores = rouge.get_scores(predictions, references, avg=True)
        return scores 

    def train_step(self, encoder_input, decoder_input, decoder_output, model, optimizer, loss_writer, acc_writer):
        with GradientTape() as tape:
            prediction = model(encoder_input, decoder_input, training=True)
            loss = self.calculate_loss(decoder_output, prediction)
            accuracy = self.calculate_accuracy(decoder_output, prediction)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(accuracy)

        # Log loss and accuracy to TensorBoard
        with loss_writer.as_default():
            tf.summary.scalar('Loss', self.train_loss.result(), step=optimizer.iterations)
        with acc_writer.as_default():
            tf.summary.scalar('Accuracy', self.train_accuracy.result(), step=optimizer.iterations)

    def train(self):
        lr_schedule = self.CustomLRSchedule(self.model_dim)
        optimizer = Adam(lr_schedule, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)

        data_preparation = PrepareDataset()
        X_train, y_train, X_valid, y_valid, X_test, y_test, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = data_preparation()

        with open('dec_tokenizer.pkl', 'rb') as f:
           dec_tokenizer = load(f)

        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(self.batch_size)
        val_data = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(self.batch_size)

        model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length,
                                 self.num_heads, self.keys_dim, self.values_dim, self.model_dim, self.feedforward_dim, self.num_layers,
                                 self.dropout_rate)

        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=None)

        train_loss_dict = {}
        val_loss_dict = {}
        train_accuracy_dict = {}
        val_accuracy_dict = {}

        train_bleu_dict = {}
        val_bleu_dict = {}
        train_rouge_dict = {}
        val_rouge_dict = {}

        # create separate writers for each metric
        train_loss_writer = tf.summary.create_file_writer('./logs/train/loss')
        train_acc_writer = tf.summary.create_file_writer('./logs/train/accuracy')
        train_bleu_writer = tf.summary.create_file_writer('./logs/train/bleu')
        train_rouge_writer = tf.summary.create_file_writer('./logs/train/rouge')

        val_loss_writer = tf.summary.create_file_writer('./logs/val/loss')
        val_acc_writer = tf.summary.create_file_writer('./logs/val/accuracy')
        val_bleu_writer = tf.summary.create_file_writer('./logs/val/bleu')
        val_rouge_writer = tf.summary.create_file_writer('./logs/val/rouge')

        for epoch in range(self.epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()

            print("Start of epoch %d" % (epoch + 1))
            start_time = time()

            for step, (train_batchX, train_batchY) in enumerate(train_data):
                encoder_input = train_batchX[:, 1:]
                decoder_input = train_batchY[:, :-1]
                decoder_output = train_batchY[:, 1:]

                self.train_step(encoder_input, decoder_input, decoder_output, model, optimizer, train_loss_writer, train_acc_writer)

                if step % 50 == 0:
                    print(f"Epoch {epoch + 1} Step {step} Loss {self.train_loss.result():.4f} "
                          + f"Accuracy {self.train_accuracy.result():.4f}")

                # Compute and log BLEU and ROUGE scores for the last batch in training data
                if step == len(train_data) - 1:
                    references = [' '.join(ref) for ref in dec_tokenizer.sequences_to_texts(train_batchY.numpy())]
                    predictions = [' '.join(pred) for pred in dec_tokenizer.sequences_to_texts(tf.argmax(model(encoder_input, decoder_input, training=False), axis=-1).numpy())]
                    bleu = self.calculate_bleu(references, predictions)
                    rouge = self.calculate_rouge(references, predictions)
                    train_bleu_dict[epoch] = bleu
                    train_rouge_dict[epoch] = rouge

                    with train_bleu_writer.as_default():
                        tf.summary.scalar('BLEU', bleu, step=epoch)
                    with train_rouge_writer.as_default():
                        tf.summary.scalar('ROUGE-L', rouge['rouge-l']['f'], step=epoch)

            for step, (val_batchX, val_batchY) in enumerate(val_data):
                encoder_input = val_batchX[:, 1:]
                decoder_input = val_batchY[:, :-1]
                decoder_output = val_batchY[:, 1:]

                prediction = model(encoder_input, decoder_input, training=False)
                loss = self.calculate_loss(decoder_output, prediction)
                accuracy = self.calculate_accuracy(decoder_output, prediction)
                self.val_loss(loss)
                self.val_accuracy(accuracy)

                # Log loss and accuracy to TensorBoard
                with val_loss_writer.as_default():
                    tf.summary.scalar('Loss', self.val_loss.result(), step=optimizer.iterations)
                with val_acc_writer.as_default():
                    tf.summary.scalar('Accuracy', self.val_accuracy.result(), step=optimizer.iterations)

                # Compute and log BLEU and ROUGE scores for the last batch in validation data
                if step == len(val_data) - 1:
                    references = [' '.join(ref) for ref in dec_tokenizer.sequences_to_texts(val_batchY.numpy())]
                    predictions = [' '.join(pred) for pred in dec_tokenizer.sequences_to_texts(tf.argmax(prediction, axis=-1).numpy())]
                    bleu = self.calculate_bleu(references, predictions)
                    rouge = self.calculate_rouge(references, predictions)
                    val_bleu_dict[epoch] = bleu
                    val_rouge_dict[epoch] = rouge

                    with val_bleu_writer.as_default():
                        tf.summary.scalar('BLEU', bleu, step=epoch)
                    with val_rouge_writer.as_default():
                        tf.summary.scalar('ROUGE-L', rouge['rouge-l']['f'], step=epoch)

            print(f"Epoch {epoch+1}: Training Loss {self.train_loss.result():.4f}, "
                  + f"Training Accuracy {self.train_accuracy.result():.4f}, "
                  + f"Validation Loss {self.val_loss.result():.4f}, "
                  + f"Validation Accuracy {self.val_accuracy.result():.4f}, "
                  + f"Training BLEU {train_bleu_dict[epoch]}, "
                  + f"Training ROUGE-L {train_rouge_dict[epoch]['rouge-l']['f']}, "
                  + f"Validation BLEU {val_bleu_dict[epoch]}, "
                  + f"Validation ROUGE-L {val_rouge_dict[epoch]['rouge-l']['f']}")

            if (epoch + 1) % 1 == 0:
                save_path = ckpt_manager.save()
                print(f"Saved checkpoint at epoch {epoch+1}\n")
                model.save_weights(f"weights/wghts{epoch + 1}.ckpt")

                train_loss_dict[epoch] = self.train_loss.result()
                val_loss_dict[epoch] = self.val_loss.result()
                train_accuracy_dict[epoch] = self.train_accuracy.result()
                val_accuracy_dict[epoch] = self.val_accuracy.result()



        print("Total time taken: %.2fs" % (time() - start_time))

        # at the end of the training save the model
        tf.keras.models.save_model(model, 'saved_model')


if __name__ == "__main__":
    train_model = TrainModel()
    train_model.train()
