import os
import sys
import pgn
from torch.utils.data import Dataset
import itertools
import torch
from tqdm import tqdm
import pickle

rows = 'abcdefgh'
cols = '12345678'


checkers_to_othello = {
    1: 'b1',
    2: 'd1',
    3: 'f1',
    4: 'h1',
    5: 'a2',
    6: 'c2',
    7: 'e2',
    8: 'g2',
    9: 'b3',
    10: 'd3',
    11: 'f3',
    12: 'h3',
    13: 'a4',
    14: 'c4',
    15: 'e4',
    16: 'g4',
    17: 'b5',
    18: 'd5',
    19: 'f5',
    20: 'h5',
    21: 'a6',
    22: 'c6',
    23: 'e6',
    24: 'g6',
    25: 'b7',
    26: 'd7',
    27: 'f7',
    28: 'h7',
    29: 'a8',
    30: 'c8',
    31: 'e8',
    32: 'g8',
}


def chartoint(pos):
    if len(pos) == 2:
        if pos[0] not in rows or pos[1] not in cols:
            return -1
        return rows.index(pos[0]) * 8 + cols.index(pos[1])
    elif pos == 'x':
        return 64
    elif pos == '|':
        return 65
    else:
        return -1
    

def inttochar(pos):
    if 0 <= pos <= 63:
        l = pos // 8
        c = pos % 8
        return rows[l] + cols[c]
    elif pos == 64:
        return 'x'
    elif pos == 65:
        return '|'
    else:
        return -1


class CheckersCharDataset(Dataset):
    def __init__(self, sequences):
        chars = sorted(list(range(64)) + [-100, ])
        chars.remove(27)
        chars.remove(28)
        chars.remove(35)
        chars.remove(36)
        chars.extend([27, 28, 35, 36, 64, 65])
        data_size, vocab_size = len(sequences), len(chars)  # vocab size 61, with -100 sorted to the front
        print('Dataset created has %d sequences, %d unique words.' % (data_size, vocab_size))
        
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        # sticking with 60 for now so that positional encodings don't get messed up 
        # but unclear if pre trained positional encodings will even be good for checkers task
        self.max_len = 60
        self.block_size = self.max_len - 1  # for autoregressive training
        self.vocab_size = vocab_size
        self.data = sequences
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx]
        if len(chunk) > self.max_len:
            chunk = chunk[:self.max_len]
        elif len(chunk) < self.max_len:
            chunk += [-100, ] * (self.max_len - len(chunk))  # -100 can be ignored in CE
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y



load_from_pickle = True
exec(open('configurator.py').read()) # overrides from command line or config file


if not load_from_pickle:
    base_dir = os.path.join(os.getcwd())

    with open(os.path.join(base_dir, 'raw_data/OCA_2.0.pdn'), 'r') as f:
        pgn_text = f.read()

    games = pgn.loads(pgn_text)

    print("Processing game data...")

    sequences = []
    # each game gets a list of moves
    # for each move, split it up into its tokens: 
    for game in tqdm(games):
        tokens = []
        try:
            for move in game.moves:
                if move in ['0-1', '1-0', '1/2-1/2']:
                    continue
                if 'x' in move:
                    temp = [checkers_to_othello[int(e)] for e in move.split('x') if e]
                    for t in temp:
                        tokens.append(t)
                        tokens.append('x')
                    tokens.pop()
                else:
                    tokens.extend([checkers_to_othello[int(e)] for e in move.split('-')])
                tokens.append('|')
            tokens = [chartoint(t) for t in tokens]
            
        except ValueError:
            continue
        sequences.append(tokens)

    print("Number of games processed: {}".format(len(sequences)))
    with open('processed_checkers.pkl', 'wb') as f:
        pickle.dump(sequences, f)


