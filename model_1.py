# Some code from https://colab.research.google.com/drive/1tYL35_0M3TobYv0eT8Uc_H4JO2JUfi4K#scrollTo=jGjHJU6w0kjy

import torch.utils
import torch
import torch.utils.data
import torch.optim as optim

import scipy.signal
from scipy.io.wavfile import write as saveWAV
import sklearn.preprocessing
import norbert                         # https://github.com/sigsep/norbert
import sounddevice as sd

import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import random

import musdb
import model

import copy
from pickle_operations import *


class SimpleMUSDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        subset='train', # select a _musdb_ subset train or test
        split='train',  # w/ subset=train, split='train' loads the training split, 'valid' loads the validation split. None applies no splitting.
        target='vocals',
        seq_duration=5.0,  # seq_duration is in seconds
    ):
        self.seq_duration = seq_duration
        self.target = target
        # musdb gives us the tracks to train/validate/test on
        # note that each track in musdb is present in exactly one of
        # the training, validation, and test sets
        self.mus = musdb.DB(
            download=True,
            split=split,
            subsets=subset, 
        )

    def __getitem__(self, index):
        track = self.mus[index]
        # Choose a random chunk of seq_duration seconds in the track
        # This lets us save some memory or compute time
        track.chunk_start = random.uniform(0, track.duration - self.seq_duration)
        track.chunk_duration = self.seq_duration
        x = track.audio.T
        # y is the seperated vocal track that we are trying to estimate
        y = track.targets[self.target].audio.T
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.mus)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

class LossStats:
    def __init__(self, num_epochs):
        self.stats = pd.DataFrame({
            'epoch': [i+1 for i in range(num_epochs)], 
            'training_loss': [-1 for i in range(num_epochs)],
            'validation_loss': [-1 for i in range(num_epochs)]})
        
    def update(self, epoch, train_loss, valid_loss):
        self.stats.at[epoch, 'training_loss'] = train_loss
        self.stats.at[epoch, 'validation_loss'] = valid_loss
    
    def to_csv(self, filepath):
        self.stats.to_csv(filepath, index=False)


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio

def findMeanScale(dataset):
    stft = model.STFT(n_fft=2048, n_hop=1024)
    spec = model.Spectrogram(mono=True)
    transform = torch.nn.Sequential(stft, spec)
    scaler = sklearn.preprocessing.StandardScaler()

    for x, y in (dataset):
        X = transform(x[None])
        scaler.partial_fit(X.squeeze().numpy())

    # set inital input scaler values
    scale = np.maximum(
        scaler.scale_,
        1e-4*np.max(scaler.scale_)
    )
    mean = scaler.mean_

    return mean, scale


def getEstimate(track_audio, device = 'cpu'):
    unmix = get_from_pickle('unmixer.pickle')
    unmix.stft.center = True
    unmix.eval()

    audio_torch = torch.tensor(track_audio.T[None, ...]).float().to(device)

    # Predict separated vocal spectrogram
    Vj = unmix(audio_torch).cpu().detach().numpy()

    # compute STFT (spectrogram of mix)
    X = unmix.stft(audio_torch).detach().cpu().numpy()
    # convert to complex numpy
    X = X[..., 0] + X[..., 1]*1j
    X = X[0].transpose(2, 1, 0)

    V = Vj[:, 0, ...].transpose(0, 2, 1)[..., None]

    #  Returns [vocals_spectrogram, accompaniment_spectrogram] using spectral subtraction
    V = norbert.residual_model(
        v=V,    # Estimated spectrogram for the vocals
        x=X     # complex mixture 
    )
    # Uses the spectrograms in V to seperate X in STFT domain into sources in STFT domain
    Y = norbert.wiener(V, X.astype(np.complex128), iterations=2)

    # Recover audio from fourier transform
    audio_hat = istft(
        Y[..., 0].T,    # 0 for vocal separation (1 for accompaniment)
        n_fft=unmix.stft.n_fft,
        n_hopsize=unmix.stft.n_hop
    )
    estimate = audio_hat.T

    return estimate


def validation_loss(valid_sampler, unmix):
    # Should be one iteration as batch size should be entire validation set
    for x, y in valid_sampler:
        x, y = x.to(device), y.to(device)

        # Our model's prediction for x
        Y_hat = unmix(x)
        # Actual vocal track spectrogram
        Y = unmix.transform(y)
        return torch.nn.functional.mse_loss(Y_hat, Y)
        print('Validation loss: {}'.format(loss))



if __name__ == '__main__':
    TRAIN = True

    torch.manual_seed(3)
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if TRAIN:
        train_dataset = SimpleMUSDBDataset(seq_duration=5.0)
        validation_set = SimpleMUSDBDataset(split = 'valid', seq_duration=5.0)
        mean, scale = findMeanScale(train_dataset)

        # why train on small batches instead of all of data at once: 
        # https://datascience.stackexchange.com/questions/16807/why-mini-batch-size-is-better-than-one-single-batch-with-all-training-data
        # we shuffle the data into new batches at every epoch
        train_sampler = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        valid_sampler = torch.utils.data.DataLoader(validation_set, batch_size=len(validation_set), shuffle=True) 
        print('Training on {} tracks'.format(len(train_dataset)))
        print('Validating on {} tracks'.format(len(validation_set)))

        unmix = model.OpenUnmix(
            input_mean=mean,
            input_scale=scale,
            nb_channels=1,
            hidden_size=256,
            n_fft=2048,
            n_hop=1024,
            max_bin=64,
            sample_rate=44100
        ).to(device)
        
        # Reading on RMSProp https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a
        optimizer = optim.RMSprop(unmix.parameters(), lr=0.001)
        # Mean squarred error is the typical error a function used for regression problems
        criterion = torch.nn.MSELoss()

        losses = AverageMeter()
        unmix.train()   # Sets module in "training mode" (this isn't a one-liner to train model :))
        best_loss = float('inf')
        model_cache = unmix

        # num_epochs is the number of times that our model gets to see the entire dataset
        num_epochs = 30
        model_stats = LossStats(num_epochs)

        for i in range(num_epochs):
            print('Starting epoch {} of {}'.format(i+1, num_epochs))
            losses.clear()
            for x, y in train_sampler:
                x, y = x.to(device), y.to(device)

                # sets the gradients to zero before starting to do backpropragation because PyTorch 
                # accumulates the gradients on subsequent backward passes.
                optimizer.zero_grad()

                # Our model's prediction for x (note that classes are callable in Python)
                Y_hat = unmix(x)
                Y = unmix.transform(y)
                loss = torch.nn.functional.mse_loss(Y_hat, Y)

                # Compute gradient of error function using back propagation
                loss.backward()
                
                # performs a parameter update based on the current gradient
                optimizer.step()
                losses.update(loss.item(), Y.size(1))

            valid_loss = validation_loss(valid_sampler, unmix)
            model_stats.update(i, losses.avg, valid_loss)
            print('Training loss: {}'.format(losses.avg))
            print('Validation loss: {}'.format(valid_loss))

            if valid_loss < best_loss:
                best_loss = valid_loss
                model_cache = copy.deepcopy(unmix)
                print('\tNew best! :D')

        model_stats.to_csv('model_stats.csv')
        save_to_pickle(model_cache, 'unmixer2.pickle')
        quit()

    musTest = musdb.DB(download=True, subsets='test')
    musTrain = musdb.DB(download=True, subsets= 'train')
    checkUnseen = True

    track = musTest[0] if checkUnseen else musTrain[0]
    print(track.name)

    estimate = getEstimate(track.audio)
    saveWAV('original_mix_111.wav', track.rate, track.audio)
    saveWAV('separated_vocals_111.wav', track.rate, estimate)

    print('Playing original track')
    sd.play(track.audio, track.rate, blocking = True)
    print('Playing estimated vocal track')
    sd.play(estimate, track.rate, blocking = True)
