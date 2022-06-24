import ast

import librosa, librosa.display

import os
import numpy as np
import pandas as pd
from tempfile import mktemp
from pydub import AudioSegment
import matplotlib.pyplot as plt


# This representation uses the method of 1 to project chroma
# features onto a 6-dimensional basis representing the perfect fifth, minor third, and major third each as
# two-dimensional coordinates.
def plot_tempogram(filename):
    signal, sr = librosa.load(filename)

    # The amount of samples we are shifting after each fft
    hop_length = 200
    # n_fft = 2048

    onset_env = librosa.onset.onset_strength(signal, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, win_length=400, hop_length=hop_length)

    plt.figure(figsize=(20, 5))
    librosa.display.specshow(tempogram, sr=sr, x_axis='time', y_axis='tempo', hop_length=hop_length)
    plt.title('Tempogram', fontdict=dict(size=18))
    plt.show()


def plot_mel_spectrogram(filename):
    signal, sr = librosa.load(filename)

    # this is the number of samples in a window per fft
    n_fft = 2048
    # The amount of samples we are shifting after each fft
    hop_length = 512
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(power_to_db, sr=sr, hop_length=hop_length)
    plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Frequency', fontdict=dict(size=15))
    plt.show()


def generate_mel_spectrogram(filename, outpath):
    signal, sr = librosa.load(filename)

    path, file = os.path.split(filename)
    pre, ext = os.path.splitext(file)
    new_filename = str(int(pre)) + ".png"
    # this is the number of samples in a window per fft
    n_fft = 2048
    # The amount of samples we are shifting after each fft
    hop_length = 512
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    #plt.figure(figsize=(20, 5))

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    plt.subplots(figsize=(600 * px, 200 * px))
    plt.axis('off')
    librosa.display.specshow(power_to_db, sr=sr, hop_length=hop_length)
    plt.savefig(outpath + new_filename, bbox_inches='tight', transparent=True)
    plt.close('all')

def generate_chroma(filename, outpath):
    signal, sr = librosa.load(filename)

    path, file = os.path.split(filename)
    pre, ext = os.path.splitext(file)
    new_filename = str(int(pre)) + ".png"
    # this is the number of samples in a window per fft
    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=signal, sr=sr, hop_length=hop_length)

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    plt.subplots(figsize=(600 * px, 200 * px))
    plt.axis('off')
    librosa.display.specshow(chroma, sr=sr, hop_length=hop_length)
    plt.savefig(outpath + new_filename, bbox_inches='tight', transparent=True)
    plt.close('all')

def plot_chroma(filename):
    signal, sr = librosa.load(filename)

    COLOR = 'white'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR

    # The amount of samples we are shifting after each fft
    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=signal, sr=sr, hop_length=hop_length)

    plt.figure(figsize=(20, 5))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, hop_length=hop_length)
    plt.title('Constant Q Chroma', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Pitch', fontdict=dict(size=15))
    plt.show()


def plot_tonnetz(filename):
    signal, sr = librosa.load(filename)

    # The amount of samples we are shifting after each fft
    hop_length = 512
    tonnetz = librosa.feature.tonnetz(y=signal, sr=sr, hop_length=hop_length)

    plt.figure(figsize=(20, 5))
    librosa.display.specshow(tonnetz, sr=sr, hop_length=hop_length)
    plt.title('Tonnetz', fontdict=dict(size=18))
    plt.show()


def load(filepath):
    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks
