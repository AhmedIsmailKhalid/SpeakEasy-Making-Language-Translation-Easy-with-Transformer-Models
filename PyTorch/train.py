import os
import pickle
from time import time
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter
from prepare_dataset import PrepareDataset
from model import Transformer
from torch.utils.data import DataLoader, TensorDataset

class TrainModel:
    def __init__(self):
        # Hyperparameters (change these as needed)
        self.num_layers = 6
        self.model_dim = 128
        self.feedforward_dim = 2048
        self.num_heads = 8
        self.dropout_rate = 0.1
        self.batch_size = 64
        self.CLIP = 1
        self.EPOCHS = 2
        self.step = 0

        # Prepare the dataset
        prepare_dataset = PrepareDataset()
        (
            self.X_train,
            self.y_train,
            self.X_valid,
            self.y_valid,
            self.X_test,
            self.y_test,
            self.enc_seq_length,
            self.dec_seq_length,
            self.enc_vocab_size,
            self.dec_vocab_size,
        ) = prepare_dataset()

        # Create the Transformer model
        self.model = Transformer(
            self.num_layers,
            self.model_dim,
            self.num_heads,
            self.feedforward_dim,
            self.enc_vocab_size,
            self.dec_vocab_size,
            self.enc_seq_length,
            self.dec_seq_length,
            self.dropout_rate,
        )

        # Save dataset parameters
        self.dataset_params = {
            "enc_seq_length": self.enc_seq_length,
            "dec_seq_length": self.dec_seq_length,
            "enc_vocab_size": self.enc_vocab_size,
            "dec_vocab_size": self.dec_vocab_size,
        }
        with open("dataset_params.pkl", "wb") as f:
            pickle.dump(self.dataset_params, f)

        # Save model parameters
        self.model_params = {
            "num_layers": self.num_layers,
            "model_dim": self.model_dim,
            "feedforward_dim": self.feedforward_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        }
        with open("model_params.pkl", "wb") as f:
            pickle.dump(self.model_params, f)

        # Choose device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to the chosen device
        self.model.to(self.device)

        # Define the optimizer and loss function
        self.optimizer = Adam(
            self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )
        self.criterion = CrossEntropyLoss(ignore_index=0)

        # Create DataLoaders for training and validation sets
        train_data = TensorDataset(self.X_train, self.y_train)
        self.train_iterator = DataLoader(train_data, batch_size=self.batch_size)

        valid_data = TensorDataset(self.X_valid, self.y_valid)
        self.valid_iterator = DataLoader(valid_data, batch_size=self.batch_size)

        # Create directories for checkpoints and weights if they do not exist
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        if not os.path.exists("weights"):
            os.makedirs("weights")

        # Create SummaryWriters for TensorBoard for both training and validation
        self.train_writer = SummaryWriter("logs/train")
        self.valid_writer = SummaryWriter("logs/valid")

    # Define the function for the learning rate scheduler
    def get_lr(self, step_num, warmup_steps=4000):
        arg1 = step_num ** -0.5
        arg2 = step_num * (warmup_steps ** -1.5)
        return (self.model_dim ** -0.5) * min(arg1, arg2)

        # Define the train_step function
    def train_step(self, mode="train", epoch=1, step_num=0):
        if mode == "train":
            self.model.train()
            iterator = self.train_iterator
            writer = self.train_writer
        else:
            self.model.eval()
            iterator = self.valid_iterator
            writer = self.valid_writer

        total_loss = 0
        total_correct = 0
        total_count = 0

        #step = 0  # Initialize step here

        for batch in iterator:
            src = batch[0].to(self.device)
            trg = batch[1].to(self.device)

            # Update learning rate only in 'train' mode
            if mode == "train":
                step_num = epoch * len(self.train_iterator) + self.step + 1
                lr = self.get_lr(step_num)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr

            self.optimizer.zero_grad()

            output = self.model(src, trg)
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = self.criterion(output, trg)
            if mode == "train":
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.CLIP)
                self.optimizer.step()

            total_loss += loss.item()

            # Compute accuracy
            pred = output.argmax(1)
            total_correct += (pred == trg).sum().item()
            total_count += len(trg)

            # Print and log every 50 steps
            if self.step % 50 == 0:
                avg_loss = total_loss / (self.step + 1)
                avg_acc = total_correct / total_count
                print(
                    f"Epoch {epoch}, Step {self.step}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}"
                )
                writer.add_scalar("Running Loss", avg_loss, epoch * len(iterator) + self.step)
                writer.add_scalar("Running Acc", avg_acc, epoch * len(iterator) + self.step)
            
            self.step += 1  # Increment step manually

        return total_loss / len(iterator), total_correct / total_count


    def train(self):
        # Train the model
        for epoch in range(1, self.EPOCHS + 1):
            print(f"Start of epoch {epoch}")
            start_time = time()
            train_loss, train_acc = self.train_step("train", epoch)
            valid_loss, valid_acc = self.train_step("eval", epoch)

            print(
                f"Epoch {epoch}:, Train Loss {train_loss:.4f}, Train Accuracy {train_acc:.4f}, Val Loss {valid_loss:.4f}, Val Accuracy {valid_acc:.4f}"
            )

            self.train_writer.add_scalar("Loss", train_loss, epoch)
            self.train_writer.add_scalar("Accuracy", train_acc, epoch)
            self.valid_writer.add_scalar("Loss", valid_loss, epoch)
            self.valid_writer.add_scalar("Accuracy", valid_acc, epoch)

            print(f"Saved checkpoint at epoch {epoch}\n")

            # Save model checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": train_loss,
                },
                f"checkpoints/model_epoch{epoch}.pth",
            )

            # Save model weights
            torch.save(
                self.model.state_dict(),
                f"weights/model_weights_epoch{epoch}.pth",
            )

        self.train_writer.close()
        self.valid_writer.close()

        # Create directories for checkpoints and weights if they do not exist
        if not os.path.exists("saved_model"):
            os.makedirs("saved_model")
        # Save model architecture
        torch.save(self.model.state_dict(), "./saved_model/saved_model.pth")

        print("Total time taken: %.2fs" % (time() - start_time))


if __name__ == "__main__":
    train_model = TrainModel()
    train_model.train()

