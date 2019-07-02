#!/usr/bin/env python3
import sys
import argparse
import logging
import h5py
import numpy as np
import librosa
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--n-basis', type=int, default=0)
    parser.add_argument('--with-deformation', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    logging.getLogger().setLevel(10 if args.verbose else 20)
    logging.debug('loading sep info from {}'.format(args.input))
    with h5py.File(args.input, 'r') as f:
        F = f['F'][()]
        D = f['D'][()]
        G = f['G'][()]
        H = f['H'][()]
        U = f['U'][()]
        Phase = f['Phase'][()]
        sr = f['sr'][()]
        n_fft = f['n_fft'][()]
        n_mel = f['n_mel'][()]
        f_min = f['f_min'][()]
        f_max = f['f_max'][()]
        channel = f['channel'][()]
        program = f['program'][()]
        note = f['note_list'][()]
    if 0 < args.n_basis < F.shape[1]:
        F = F[:, :args.n_basis]
        D = D[:, :args.n_basis]
        G = G[:args.n_basis, :]

    plt.imshow(F, origin='lower', aspect='auto')
    plt.xlabel('Note')
    plt.ylabel('Mel bin')
    plt.savefig('F.png', dpi=96)

    plt.imshow(D, origin='lower', aspect='auto')
    plt.xlabel('Note')
    plt.ylabel('Mel bin')
    plt.savefig('D.png', dpi=96)

    plt.imshow(G, origin='lower', aspect='auto')
    plt.xlabel('Time frame')
    plt.ylabel('Note')
    plt.savefig('G.png', dpi=96)

    plt.imshow(H, origin='lower', aspect='auto')
    plt.xlabel('Basis')
    plt.ylabel('Mel bin')
    plt.savefig('H.png', dpi=96)

    plt.imshow(U, origin='lower', aspect='auto')
    plt.xlabel('Time frame')
    plt.ylabel('Basis')
    plt.savefig('U.png', dpi=96)

    plt.imshow(F+D, origin='lower', aspect='auto')
    plt.xlabel('Note')
    plt.ylabel('Mel bin')
    plt.savefig('FD.png', dpi=96)

    plt.imshow(np.dot(F+D, G), origin='lower', aspect='auto')
    plt.xlabel('Time frame')
    plt.ylabel('Mel bin')
    plt.savefig('FDG.png', dpi=96)

    plt.imshow(np.dot(H, U), origin='lower', aspect='auto')
    plt.xlabel('Time frame')
    plt.ylabel('Mel bin')
    plt.savefig('HU.png', dpi=96)

    plt.imshow(np.dot(F+D, G)+np.dot(H, U), origin='lower', aspect='auto')
    plt.xlabel('Time frame')
    plt.ylabel('Mel bin')
    plt.savefig('R.png', dpi=96)

    if args.with_deformation:
        R = np.dot(F+D, G)
    else:
        R = np.dot(F, G)
    if n_mel > 0:
        mel_basis = librosa.filters.mel(sr, n_fft, n_mel, f_min, f_max)
        mel_basis_inv = np.linalg.pinv(mel_basis)
        R = np.dot(mel_basis_inv, R)
    x = librosa.istft(Phase * R)
    librosa.output.write_wav(args.output, x, sr)

if __name__ == '__main__':
    main()
