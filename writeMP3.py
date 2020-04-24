import librosa
import numpy as np
import pydub

# taken from 
# https://stackoverflow.com/questions/53633177/how-to-read-a-mp3-audio-file-into-a-numpy-array-save-a-numpy-array-to-mp3

"""
numpy array to MP3
"""
def writeMP3(filename, x, sample_rate = 44100, normalized=True):
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)

    song = pydub.AudioSegment(y.tobytes(), frame_rate=sample_rate, sample_width=2, channels=channels)
    song.export(filename, format="mp3", bitrate="320k")

if __name__ == '__main__':
    sample_rate = 44100
    inPath = '/Users/xavierboudreau/Music/iTunes/iTunes Media/Music/Drake/Take Care (Deluxe Version)/10 Make Me Proud (feat. Nicki Minaj).m4a'
    outPath = 'drizzyDrake.mp3'
    audio, _ = librosa.load(inPath, sr = sample_rate, mono = False)
    writeMP3(outPath, audio)