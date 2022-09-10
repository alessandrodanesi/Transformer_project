import logging
import math
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from functools import partial
import click
from pathlib import Path
import logging
import re
from datamaestro import prepare_dataset

from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import datetime
from pathlib import Path

from utils import *

class DeLight_Classifier(nn.Module):
    """
    DeLight Classifier model for sentiment analysis
    """
    def __init__(self, embedding_size,max_len, scaling, Nb, Wb, glt_shuffle):
        """
        :param embedding_size : dimension of the data
        :param scaling: Boolean whether choosing Block-wise scaling or not
        :param Nb : List of the number of layers in the DeLight blocks
        :param Wb : List of the width expansion ratio in the DeLight blocks
        
        """
        super(DeLight_Classifier, self).__init__()
        
        self.scaling = scaling
        self.Nb = Nb
        self.Wb = Wb
        
        self.pos_embed = PositionalEncoding(embedding_size, max_len)
        
        if scaling:
            self.F = [DeLightBlock(embedding_size, nb=Nb[i], wb=Wb[i], glt_shuffle=glt_shuffle) for i in len(Nb)]
        else:
            self.F1 = DeLightBlock(embedding_size, nb=4, wb=2.0, glt_shuffle=glt_shuffle)
            self.F2 = DeLightBlock(embedding_size, nb=4, wb=2.0, glt_shuffle=glt_shuffle)
            self.F3 = DeLightBlock(embedding_size, nb=4, wb=2.0, glt_shuffle=glt_shuffle)
            
        self.final = nn.Linear(embedding_size, 2)

    def forward(self, x):
        x = self.pos_embed(x)
        
        if self.scaling:
            for D_block in self.F:
                x = D_block(x)
        else:
            x = self.F1(x)
            x = self.F2(x)
            x = self.F3(x)
        
        x = torch.mean(x, dim=0)
        x = self.final(x)

        return x

    
def compute_Nb_wb(N_max, N_min, Nr_blocks, w):
    """
    Function to compute Block-wise scaling parameters
    
    :param N_max : Minimum number of layers in DeLight blocks
    :param N_max : Maximum number of layers in DeLight blocks
    :param Nr_blocks: Number of Delight blocks in the network
    :param w : Maximum width exmpansion ratio in DeLight block
    """
    b = np.arange(0, Nr_blocks)
    N_b = N_min + ((N_max - N_min) / (Nr_blocks - 1)) * b
    w_b = w + ((N_max - N_min) / (N_min * (Nr_blocks - 1))) * b
    
    return N_b, w_b


class State:
    """
    Checkpoint class to save the model parameters and training state
    """
    def __init__(self, model, optim, criterion):
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.epoch, self.iteration = 0, 0


class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text(encoding='utf-8') if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text()), self.filelabels[ix]

    def collate(batch):
        data = []
        Y = []
        for (x, y) in batch:
            X = []
            for i in range(len(x)):
                X.append(np.array(emb[x[i]]))

            data.append(torch.tensor(X)[:, 0, :])
            Y.append(y)

        return torch.nn.utils.rnn.pad_sequence(data), torch.LongTensor(Y)


def get_imdb_data(embedding_size=100):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()

    OOVID = len(words)
    words.append("__OOV__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=True), FolderText(
        ds.test.classes, ds.test.path, tokenizer, load=True)

if __name__ == "__main__":
    # Choose computing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training parameters
    BATCH_SIZE = 16
    NB_EPOCHS = 10
    lr = .00005
    embedding_size = 100

    # Model parameters
    N_max = 8  # n. max of GLT layers among all blocks
    N_min = 4  # n. min of GLT layers among all blocks
    Nr_DeLight_blocks = 3  # Number DeLight blocks
    width_multiplier = 2.0  # Width Epansion ratio
    N_b, w_b = compute_Nb_wb(N_max, N_min, Nr_DeLight_blocks, width_multiplier)
    N_b = N_b.astype(int)

    scaling = False
    glt_shuffle = True

    # Build Dataset
    word, emb, fold_train, fold_test = get_imdb_data(embedding_size)
    data_train = DataLoader(fold_train, shuffle=True, batch_size=BATCH_SIZE, collate_fn=FolderText.collate)
    data_test = DataLoader(fold_test, shuffle=True, batch_size=BATCH_SIZE, collate_fn=FolderText.collate)

    # Tools
    model_id = 2000000
    path_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    params_str = f'batch{BATCH_SIZE}_epoch{NB_EPOCHS}_lr{lr}'
    writer = SummaryWriter(f'runs/{model_id}/{params_str}/{path_time}_notscaling_shuffle')

    # Build Model
    savepath = Path(f'models/{model_id}_{params_str}_notscaling_shuffle.pch')

    if savepath.is_file():
        with savepath.open("rb") as fp:
            state = torch.load(fp)
            state.model.to(device=device)
    else:
        model = DeLight_Classifier(embedding_size,max_len=2470, scaling=scaling, Nb=N_b, Wb=w_b, glt_shuffle=glt_shuffle)
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(params=model.parameters(), lr=lr)
        state = State(model, optim, criterion)
    
    # Start Training
    logging.info("Training ...")

    # Starting time
    start = time.time()

    for epoch in tqdm(range(state.epoch, NB_EPOCHS)):

        Losses_train = []
        Losses_test = []
        state.model.train()

        for i,(X, y) in enumerate(data_train):
            if i%100 ==0:
                print(f'training, batch {i}')
            X = X.to(device)
            y = y.to(device)

            X = X.float()
            y_hat = state.model(X)

            loss = state.criterion(y_hat, y.long())

            loss.backward()

            state.optim.step()

            state.optim.zero_grad()

            state.iteration += 1

            Losses_train.append(loss.item())

        loss_train = sum(Losses_train) / len(Losses_train)
        writer.add_scalar('Loss/train', loss_train, state.epoch)
        print('Train loss: %f' % (loss_train))

        # Accuracy variables
        total_len = 0
        correct_preds = 0
        
        # Testing
        with torch.no_grad():
            state.model.eval()
            total = 0
            correct = 0
            for (X, y) in data_test:
                X = X.to(device)
                y = y.to(device)
                y_hat = state.model(X.float())

                loss = state.criterion(y_hat, y.long())

                Losses_test.append(loss.item())

                y_pred = torch.argmax(y_hat, dim=-1)
                assert y.shape == y_pred.shape
                correct_preds += (y_pred == y).sum().item()
                total_len += y.size(0)

        loss_test = sum(Losses_test) / len(Losses_test)
        writer.add_scalar('Loss/test', loss_test, state.epoch)

        acc = correct_preds * 100. / total_len
        writer.add_scalar('Acc/test', acc, state.epoch)

        print('Test accuracy: %4.3f, Test loss: %f' % (acc, loss_test))

        # Checkpoint
        with savepath.open("wb") as fp:
            state.epoch = epoch + 1
            torch.save(state, fp)

    # Finish time
    finish = time.time()
    
    print(finish-start)
    
    # Display Training time
    logging.info(f'Training Time is {finish-start} s')