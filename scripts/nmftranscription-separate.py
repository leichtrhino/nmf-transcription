#!/usr/bin/env python3
import sys
import argparse
import logging
import h5py
import numpy as np
import librosa
from itertools import product

def cost(Y, Rs, params):
    F, D, G, H, U = Rs
    _, mu1, mu2, mu3, mu4 = params
    B = F + D
    R = np.dot(B, G) + np.dot(H, U)

    d = np.sum(-Y*np.log(R) + R + Y*np.log(Y) - Y)
    p1 = mu1*np.linalg.norm(np.dot(F.T, D))**2
    p2 = mu2*np.linalg.norm(np.dot(F.T, H))**2
    p3 = mu3*np.linalg.norm(np.dot(D.T, H))**2
    p4 = mu4*np.linalg.norm(np.dot(B.T, H))**2
    return (d+p1+p2+p3+p4), d, p1, p2, p3, p4

def fit_step(Y, RS, params):
    F, D, G, H, U = RS
    eta, mu1, mu2, mu3, mu4 = params
    eps = 1e-30

    # update D
    B = F + D
    R = np.dot(B, G) + np.dot(H, U)
    V = 2*mu1*np.dot(np.dot(F, F.T), D) + 2*mu3*np.dot(np.dot(H, H.T), D)
    Nom = np.dot(Y/np.maximum(R, eps), G.T)
    Den = np.sum(G, axis=1) + 2*mu4*np.dot(np.dot(H, H.T), B)
    D[:] = (eta*F + D) * np.where(
        V >= 0,
        Nom/np.maximum(Den+V, eps),
        (Nom-V)/np.maximum(Den, eps)
    ) - eta*F
    # update H
    B = F + D
    R = np.dot(B, G) + np.dot(H, U)
    W = 2*mu3*np.dot(np.dot(D, D.T), H)
    S = 2*mu4*np.dot(np.dot(B, B.T), H)
    Nom = np.dot(Y/np.maximum(R, eps), U.T)
    Den = np.sum(U, axis=1) + 2*mu2*np.dot(np.dot(F, F.T), H) + S
    H[:] = H * np.where(
        W >= 0,
        Nom/np.maximum(Den+W, eps),
        (Nom-W)/np.maximum(Den, eps)
    )
    # update G
    B = F + D
    R = np.dot(B, G) + np.dot(H, U)
    G[:] = G*np.dot(B.T, Y/np.maximum(R, eps)) / np.sum(B, axis=0)[:, None]
    # update U
    B = F + D
    R = np.dot(B, G) + np.dot(H, U)
    U[:] = U*np.dot(H.T, Y/np.maximum(R, eps)) / np.sum(H, axis=0)[:, None]

    return F, D, G, H, U

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-b', '--basis', type=str, required=True)
    parser.add_argument('-L', type=int, default=30)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--mu1', type=float, default=10)
    parser.add_argument('--mu2', type=float, default=10)
    parser.add_argument('--mu3', type=float, default=10)
    parser.add_argument('--mu4', type=float, default=10)
    parser.add_argument('--max-iter', type=int, default=10000)
    parser.add_argument('--tol', type=float, default=1e-18)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    logging.getLogger().setLevel(10 if args.verbose else 20)
    logging.debug('loading basis from {}'.format(args.basis))
    with h5py.File(args.basis, 'r') as f:
        F = f['F'][()]
        sr = f['sr'][()]
        n_fft = f['n_fft'][()]
        n_mel = f['n_mel'][()]
        f_min = f['f_min'][()]
        f_max = f['f_max'][()]
        channel = f['channel'][()]
        program = f['program'][()]
        note_list = f['note_list'][()]

    logging.debug('loading audio {}'.format(args.input))
    y, _ = librosa.core.load(args.input, sr=sr)
    Y, Phase = librosa.core.magphase(librosa.core.stft(y, n_fft=n_fft))
    if n_mel > 0:
        mel_basis = librosa.filters.mel(sr, n_fft, n_mel, f_min, f_max)
        Y = np.dot(mel_basis, Y)

    T, (Omega, K), L = Y.shape[1], F.shape, args.L
    D = np.abs(np.random.randn(Omega, K))
    G = np.abs(np.random.randn(K, T))
    H = np.abs(np.random.randn(Omega, L))
    U = np.abs(np.random.randn(L, T))
    Rs = (F, D, G, H, U)
    params = (args.eta, args.mu1, args.mu2, args.mu3, args.mu4)

    cost_at_init = cost(Y, Rs, params)[0]
    cost_ = cost_at_init
    for n_iter in range(1, args.max_iter + 1):
        Rs = fit_step(Y, Rs, params)
        cost_tuple = cost(Y, Rs, params)
        new_cost, cost_components = cost_tuple[0], cost_tuple[1:]
        previous_cost, cost_ = cost_, new_cost
        logging.debug(
            'iter {} cost {:.3f}='.format(n_iter, cost_)\
            + '+'.join(map('{:.3f}'.format, cost_components))
        )
        if 0 < (previous_cost - cost_) / cost_at_init < args.tol:
            break

    with h5py.File(args.output, 'w') as f:
        f.create_dataset('F', data=F)
        f.create_dataset('D', data=D)
        f.create_dataset('G', data=G)
        f.create_dataset('H', data=H)
        f.create_dataset('U', data=U)
        f.create_dataset('Phase', data=Phase)
        f.create_dataset('channel', data=channel)
        f.create_dataset('program', data=program)
        f.create_dataset('note_list', data=note_list)
        f.create_dataset('sr', data=sr)
        f.create_dataset('n_fft', data=n_fft)
        f.create_dataset('n_mel', data=n_mel)
        f.create_dataset('f_min', data=f_min)
        f.create_dataset('f_max', data=f_max)

if __name__ == '__main__':
    main()
