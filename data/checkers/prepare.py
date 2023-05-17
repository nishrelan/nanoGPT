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
    32: 'b1',
    31: 'd1',
    30: 'f1',
    29: 'h1',
    28: 'a2',
    27: 'c2',
    26: 'e2',
    25: 'g2',
    24: 'b3',
    23: 'd3',
    22: 'f3',
    21: 'h3',
    20: 'a4',
    19: 'c4',
    18: 'e4',
    17: 'g4',
    16: 'b5',
    15: 'd5',
    14: 'f5',
    13: 'h5',
    12: 'a6',
    11: 'c6',
    10: 'e6',
    9: 'g6',
    8: 'b7',
    7: 'd7',
    6: 'f7',
    5: 'h7',
    4: 'a8',
    3: 'c8',
    2: 'e8',
    1: 'g8',
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





# only execute this part if running this file, ie if 
# we want to generate dataset
if __name__ == '__main__':
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
                
                if move.count('x') == 0:
                    tokens.extend([checkers_to_othello[int(e)] for e in move.split('-')])
                elif move.count('x') == 1:
                    tokens.extend([checkers_to_othello[int(e)] for e in move.split('x')])
                else:
                    temp = [checkers_to_othello[int(e)] for e in move.split('x')]
                    temp = [a for b in [[e, e] for e in temp] for a in b][1:-1]
                    tokens.extend(temp)
            tokens = [chartoint(t) for t in tokens]
        except ValueError:
            continue
        sequences.append(tokens)



    print("Number of games processed: {}".format(len(sequences)))
    with open('processed_checkers.pkl', 'wb') as f:
        pickle.dump(sequences, f)


