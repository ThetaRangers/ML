import librosa, librosa.display

import numpy as np
from tempfile import mktemp
from pydub import AudioSegment
import matplotlib.pyplot as plt

def plot_tempogram(filename):
    mp3_audio = AudioSegment.from_file(filename, format="mp3").set_channels(1)  # read mp3

    wname = mktemp('.wav')  # use temporary file
    mp3_audio.export(wname, format="wav")  # convert to wav
    signal, sr = librosa.load(wname)

    # The amount of samples we are shifting after each fft
    hop_length = 512
    mel_signal = librosa.feature.tempogram(y=signal, sr=sr)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', hop_length=hop_length)
    plt.title('Tempogram', fontdict=dict(size=18))
    plt.show()

def plot_mel_spectrogram(filename):
    mp3_audio = AudioSegment.from_file(filename, format="mp3").set_channels(1)  # read mp3

    wname = mktemp('.wav')  # use temporary file
    mp3_audio.export(wname, format="wav")  # convert to wav
    signal, sr = librosa.load(wname)

    # this is the number of samples in a window per fft
    n_fft = 2048
    # The amount of samples we are shifting after each fft
    hop_length = 512
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', hop_length=hop_length)
    plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Frequency', fontdict=dict(size=15))
    plt.show()

def plot_chroma(filename):
    mp3_audio = AudioSegment.from_file(filename, format="mp3").set_channels(1)  # read mp3

    wname = mktemp('.wav')  # use temporary file
    mp3_audio.export(wname, format="wav")  # convert to wav
    signal, sr = librosa.load(wname)

    # The amount of samples we are shifting after each fft
    hop_length = 512
    mel_signal = librosa.feature.chroma_cqt(y=signal, sr=sr, hop_length=hop_length)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', hop_length=hop_length)
    plt.title('Constant Q Chroma', fontdict=dict(size=18))
    plt.show()

def plot_tonnetz(filename):
    mp3_audio = AudioSegment.from_file(filename, format="mp3").set_channels(1)  # read mp3

    wname = mktemp('.wav')  # use temporary file
    mp3_audio.export(wname, format="wav")  # convert to wav
    signal, sr = librosa.load(wname)

    # The amount of samples we are shifting after each fft
    hop_length = 512
    mel_signal = librosa.feature.tonnetz(y=signal, sr=sr, hop_length=hop_length)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', hop_length=hop_length)
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