import torch.utils
import torch
import torch.utils.data
import scipy.signal
from scipy.io.wavfile import write as saveWAV
import numpy as np

import norbert                         # https://github.com/sigsep/norbert
import sounddevice as sd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import musdb
from pickle_operations import *

# inverse short time fourier transform
def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio


def getEstimate(track_audio, model_file = 'unmixer2.pickle', device = 'cpu'):
    unmix = get_from_pickle(model_file)
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



if __name__ == '__main__':
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