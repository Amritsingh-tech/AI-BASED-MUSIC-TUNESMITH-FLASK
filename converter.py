import librosa
import numpy as np
from midiutil import MIDIFile

def convert_to_midi(audio_path, midi_path):
    y, sr = librosa.load(audio_path)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    mf = MIDIFile(1)
    track = 0
    time = 0
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, tempo)
    channel = 0
    volume = 100
    for onset in onset_times:
        pitch = 64
        duration = 1
        mf.addNote(track, channel, pitch, onset, duration, volume)
    with open(midi_path, 'wb') as outf:
        mf.writeFile(outf)

if __name__ == '__main__':
    convert_to_midi(r'static\uploads\music\audio.wav', 'audio.mid')