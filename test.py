import torch.utils
import torch
import torch.utils.data
import scipy.signal
from scipy.io.wavfile import write as writeWAV
from writeMP3 import writeMP3
import numpy as np
import random
random.seed(7)

import norbert                         # https://github.com/sigsep/norbert
import sounddevice as sd
import librosa

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import musdb
from pickle_operations import *
import sys

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


def getEstimate(track_audio, model_file = 'unmixer.pickle', device = 'cpu'):
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


'''
TODO: Add optional saving and place function into class along with testLocal. Models are then loaded from state
'''
def testMUSDB():
    checkUnseen = True

    musTracks = musdb.DB(download=True, subsets='test' if checkUnseen else 'train')
    track = musTracks[7]
    print(track.name)

    estimate = getEstimate(track.audio, 'unmixer2.pickle')
    estimateBad = getEstimate(track.audio, 'unmixer.pickle')
    saveWAV('original_mix_111.wav', track.rate, track.audio)
    saveWAV('separated_vocals_111.wav', track.rate, estimate)

    print(track.rate)
    print(track.audio.shape)

    print('Playing original track')
    sd.play(track.audio, track.rate, blocking = True)
    print('Playing estimated vocal track')
    sd.play(estimate, track.rate, blocking = True)
    print('Playing poor estimated vocal track')
    sd.play(estimateBad, track.rate, blocking=True)

# Grabs a sequence of length @duration seconds from @audio with @sample_rate
def grabSequence(audio, sample_rate, duration = 10, grabRandom = True):
    total_samples = audio.shape[0]
    seq_samples = duration*sample_rate
    if seq_samples >= total_samples:
        print('ERR: Requested sequence duration is longer than track. Defaulting to entire track.')
        return audio
    
    start = random.randint(0, total_samples-seq_samples) if grabRandom else 0
    return audio[start:start+seq_samples, ...]
    


def testLocal(filePath, savePath = '', modelPath = 'intermediate_models/Production Unmixer.pickle', sample_rate = 44100, playWhenDone = False):
    audio, _ = librosa.load(filePath, sr = sample_rate, mono = False)

    # if audio is Mono
    if len(audio.shape) == 1:
        audio = audio.reshape(audio.shape[0], 1)
    else:
        audio = audio.T

    audio = grabSequence(audio, sample_rate)
    estimate = getEstimate(audio, modelPath) if modelPath else audio

    if savePath:
        if ('.mp3' in savePath[-4:].lower()):
            writeMP3(savePath, estimate, sample_rate)
        elif ('.wav' in savePath[-4:].lower()):
            writeWAV(savePath, sample_rate, estimate)
        else:
            print('File extension in SAVEPATH not supported. Trying ending with .wav or .mp3')
    
    if playWhenDone:
        sd.play(estimate, sample_rate, blocking=True)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Must run as:\npython test.py FILEPATH SAVEPATH MODELPATH')
        
    else:
        testLocal(sys.argv[1], savePath = sys.argv[2], modelPath = sys.argv[3])