from music_plots import *

import os.path
from tqdm import tqdm
from pydub import AudioSegment
from pathlib import Path

files = os.listdir('data/tracks_wav')
Path("data/spectrograms").mkdir(parents=True, exist_ok=True)
#files = files[:100]

if __name__ == "__main__":
    for file in tqdm(files):
        generate_mel_spectrogram("data/tracks_wav/" + file, "data/spectrograms/")
    """
    for file in tqdm(files):
        try:
            mp3_audio = AudioSegment.from_file("data/tracks/" + file, format="mp3").set_channels(1)
            pre, ext = os.path.splitext(file)
            new_filename = "data/tracks_wav/" + pre + ".png"
            file_wav = open(new_filename, 'w')
            mp3_audio.export(new_filename, format="wav")  # convert to wav
        except Exception as e:
            # If ffmpeg couldn't decode, remove the mp3 file
            print("Removing " + file + " Cause=" + e.__str__())
            os.remove("data/tracks/" + file)
    """