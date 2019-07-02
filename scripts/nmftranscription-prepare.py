#!/usr/bin/env python3
import argparse
import logging
import os
import re
import subprocess
import tempfile

import h5py
import numpy as np

import librosa
#from nmftranscription import inst_dict
from nmftranscription.inst_dict import melodic_dict, percussion_dict
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

def make_training_audio(channel, program, note_list, duration, sr):
    with tempfile.TemporaryDirectory() as tmp_dir:
        midi_path = os.path.join(tmp_dir, 'train.mid')
        wave_path = os.path.join(tmp_dir, 'train.wav')
        fnull = open(os.devnull, 'w')

        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=bpm2tempo(240)))
        track.append(Message('program_change', channel=channel-1, program=program-1, time=0))
        for ni, note in enumerate(note_list):
            track.append(Message('note_on', note=note, time=int(1000*duration) if ni > 0 else 0))
            track.append(Message('note_off', note=note, time=int(1000*duration)))
        mid.save(midi_path)

        subprocess.call(
            ['musescore', '-o', wave_path, midi_path],
            stdout=fnull, stderr=fnull
        )
        y, _ = librosa.core.load(wave_path, sr, duration=(len(note_list)+1)*duration)
    return y

def fit(Y, n_basis, n_iter, eps=1e-30):
    F = np.abs(np.random.randn(Y.shape[0], n_basis))
    Q = np.abs(np.random.randn(n_basis, Y.shape[1]))
    def div_kl():
        Yhat = np.dot(F, Q)
        return np.sum(Y*np.log(np.maximum(Y, eps)/np.maximum(Yhat, eps))-Y+Yhat)
    for i in range(1, n_iter+1):
        F[:] = F*np.dot(Y/np.maximum(np.dot(F, Q), eps), Q.T)\
          / np.maximum(np.sum(Q, axis=1), eps)
        Q[:] = Q*np.dot(F.T, Y/np.maximum(np.dot(F, Q), eps))\
          / np.maximum(np.sum(F, axis=0), eps)[:, None]
        logging.debug('iter {} cost {:.3f}'.format(i, div_kl()))
    return F, Q

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--n-fft', type=int, default=2048)
    parser.add_argument('--n-mel', type=int, default=128)
    parser.add_argument('--f-min', type=float, default=0)
    parser.add_argument('--f-max', type=float, default=None)
    parser.add_argument('--auto-freq', action='store_true')
    parser.add_argument('--n-basis', type=int, default=0)
    parser.add_argument('--n-iter', type=int, default=1000)
    parser.add_argument('--duration', type=float, default=1.)
    parser.add_argument('--melodic', type=str)
    parser.add_argument('--melodic-num', type=int, default=0)
    parser.add_argument('--note-range', nargs=2, default=('C4', 'D#5'))
    parser.add_argument('--note-num-list', nargs='*')
    parser.add_argument('--percussive', type=str)
    parser.add_argument('--percussive-num', type=int, default=0)
    parser.add_argument('-v', '--verbose', action='store_true')

    note_regex = re.compile(r'([A-G][#b]?)([0-9])')
    note_offset = {
        'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
        'C': 0, 'C#':  1, 'Db':  1, 'D':  2, 'D#': 3, 'Eb': 3, 'E': 4,
        'F': 5, 'F#':  6, 'Gb':  6, 'G':  7, 'G#': 8, 'Ab': 8
    }
    def to_midi_note(s):
        a, b = note_regex.match(s).groups()
        return 60 + note_offset[a] + 12*(int(b)-4)
    to_freq = lambda n: 2**((n-69)/12)*440

    args = parser.parse_args()
    if not args.melodic_num and args.melodic:
        args.melodic_num, _ = melodic_dict[args.melodic]
    if not args.percussive_num and args.percussive:
        args.percussive_num, _ = percussion_dict[args.percussive]
    assert(args.melodic_num or args.percussive_num)
    assert(not args.melodic_num or not args.percussive_num)
    if args.melodic_num:
        if not args.note_num_list:
            nmin, nmax = tuple(map(to_midi_note, args.note_range))
            args.note_num_list = list(range(nmin, nmax+1))
        args.note_num_list = sorted(args.note_num_list)
        if args.auto_freq:
            args.f_min = to_freq(args.note_num_list[0]) * 0.8
            args.f_max = max(4*to_freq(args.note_num_list[-1])*1.2, args.sr/2)
    elif args.percussive_num:
        args.note_num_list = [args.percussive_num]
    if not args.f_max:
        args.f_max = args.sr / 2
    if not args.n_basis:
        args.n_basis = len(args.note_num_list)
    return args

def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.melodic_num:
        channel, program = 1, args.melodic_num
    elif args.percussive_num:
        channel, program = 10, 1
    note_list = args.note_num_list
    
    y = make_training_audio(
        channel, program, note_list, args.duration, args.sr
    )
    Y, _ = librosa.magphase(librosa.core.stft(y, args.n_fft))

    if args.n_mel > 0:
        mel_basis = librosa.filters.mel(
            args.sr, args.n_fft, args.n_mel, args.f_min, args.f_max
        )
        Y = np.dot(mel_basis, Y)

    F, Q = fit(Y, args.n_basis, args.n_iter, eps=1e-30)
    F[:] = F[:, np.argsort(np.argmax(F, axis=0))]

    with h5py.File(args.output, 'w') as f:
        f.create_dataset('F', data=F)
        f.create_dataset('channel', data=channel)
        f.create_dataset('program', data=program)
        f.create_dataset('note_list', data=note_list)
        f.create_dataset('sr', data=args.sr)
        f.create_dataset('n_fft', data=args.n_fft)
        f.create_dataset('n_mel', data=args.n_mel)
        f.create_dataset('f_min', data=args.f_min)
        f.create_dataset('f_max', data=args.f_max)
        f.create_dataset('n_basis', data=args.n_basis)

if __name__ == '__main__':
    main()
