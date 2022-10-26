#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:33:21 2022

@author: giridhar
"""

from tensorflow.python.platform import tf_logging as logging
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.utils import Sequence
import numpy as np
from scipy.io import wavfile
import random
import scipy
_,H = scipy.signal.freqz([1,-0.97],1, worN=512, whole=1)


def mfcc(wav, frame_length=1024, frame_step=256, fft_length=1024, num_mels=80, num_mfccs=60):
    n_freq = fft_length//2
    mel_wt= tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mels, num_spectrogram_bins=n_freq+1, 
        sample_rate=16000,lower_edge_hertz=0., upper_edge_hertz=8000., 
        dtype=tf.dtypes.float32, name=None)
    
    spec = tf.signal.stft(wav,frame_length=frame_length, 
            frame_step=frame_step, fft_length=fft_length, pad_end=True)
    
    mag_spec = tf.math.abs(spec)
    mel_spec=tf.tensordot(mag_spec**2, mel_wt,1)
    log_mel_spectrograms = tf.math.log(mel_spec + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :60]
    return mfccs


def frame_exc(exc, frame_length, frame_shift):
    framed_exc = tf.signal.frame(exc, frame_length, frame_shift, pad_end=True, pad_value=0, axis=0)
    hann_wind = tf.signal.hann_window(frame_length, periodic=True, dtype=tf.dtypes.float32)
    framed_exc = framed_exc*hann_wind
    return framed_exc

def frame_exct(exc, frame_length, frame_shift):
    framed_exc = tf.signal.frame(exc, frame_length, frame_shift, pad_end=True, pad_value=0, axis=0)
    return framed_exc

def preemphasis(wav, coef=0.97):
    wav=tf.pad(wav, paddings=[[0,0],[1,0]])
    wav = wav[:, 1:] - coef * wav[:, :-1]
    return wav


class dg_nlvocoder(Sequence):
    def __init__(self, speech_file, exc_file, batch_size=8):
        self.batch_size = batch_size
        self.nfft = 1024
        self.sp_size = int(self.nfft/2) + 1
        self.mfcc_size = 60
        self.seg_length = 32000
        self.frame_shift = 80
        self.frame_length = 400
        self.fs, self.speech = wavfile.read(speech_file)
        self.speech = np.append(self.speech, np.zeros([(self.seg_length-(len(self.speech)%self.seg_length))]))
        self.fs, self.exct = wavfile.read(exc_file)
        self.exct = np.append(self.exct, np.zeros([(self.seg_length-(len(self.exct)%self.seg_length))])).astype(np.float32)
        self.speech = self.speech[:len(self.exct)]
        #self.exct = self.exct[:-960000]
        #self.speech = self.speech[:-960000]
        self.n_frames = int(self.seg_length/self.frame_shift)
        self.L = self.speech.shape[0]
        self.X = np.zeros((self.batch_size, self.seg_length+320), dtype=np.float32)
        self.spch = np.zeros((self.batch_size, self.seg_length), dtype=np.float32)
        self.E = np.zeros((self.batch_size, self.n_frames, self.frame_length), dtype=np.float32)
        self.MFCC = np.zeros((self.batch_size, self.n_frames, self.mfcc_size), dtype=np.float32)
        self.idx = np.arange(0, self.L - 2*self.seg_length, self.seg_length)
        self.B = len(self.idx) // self.batch_size       
        self.shuffle=True
        self.on_epoch_end()

    def __len__(self):
        return self.B
    
    def on_epoch_end(self):
        if self.shuffle:
            seed = np.random.randint(0, self.L)
            self.idx = np.concatenate([
                np.flip(np.arange(seed, 0, -self.seg_length)),
                np.arange(seed+self.seg_length, self.L, self.seg_length)
                ])
            self.idx = self.idx[:-1]
            random.shuffle(self.idx)

    def __getitem__(self, index):
       
        for i in range(self.batch_size):
            k = index*self.batch_size + i
            left = self.idx[k]
            right = left+self.seg_length
            self.spch[i] = self.speech[left:right]*1./(2 ** 15 - 1)
            exc = self.exct[left:right]*1./(2 ** 15)
            self.E[i] = frame_exct(exc, self.frame_length, self.frame_shift)
            self.X[i] = np.append(self.spch[i], np.zeros(320))
        M = mfcc(self.spch, frame_length=400, frame_step=80, fft_length=1024, num_mels=80, num_mfccs=60)
        return [M, self.E], [self.X, self.X, self.X, self.X] 


class dg_nlvocoder_256(Sequence):
    def __init__(self, speech_file, exc_file, batch_size=8):
        self.batch_size = batch_size
        self.nfft = 1024
        self.sp_size = int(self.nfft/2) + 1
        self.mfcc_size = 60
        self.seg_length = 32000
        self.frame_shift = 80
        self.frame_length = 256
        self.fs, self.speech = wavfile.read(speech_file)
        self.speech = np.append(self.speech, np.zeros([(self.seg_length-(len(self.speech)%self.seg_length))]))
        self.fs, self.exct = wavfile.read(exc_file)
        self.exct = np.append(self.exct, np.zeros([(self.seg_length-(len(self.exct)%self.seg_length))])).astype(np.float32)
        self.speech = self.speech[:len(self.exct)]
        #self.exct = self.exct[:-960000]
        #self.speech = self.speech[:-960000]
        self.n_frames = int(self.seg_length/self.frame_shift)
        self.L = self.speech.shape[0]
        self.X = np.zeros((self.batch_size, self.seg_length+176), dtype=np.float32)
        self.spch = np.zeros((self.batch_size, self.seg_length), dtype=np.float32)
        self.E = np.zeros((self.batch_size, self.n_frames, self.frame_length), dtype=np.float32)
        self.MFCC = np.zeros((self.batch_size, self.n_frames, self.mfcc_size), dtype=np.float32)
        self.idx = np.arange(0, self.L - 2*self.seg_length, self.seg_length)
        self.B = len(self.idx) // self.batch_size       
        self.shuffle=True
        self.on_epoch_end()

    def __len__(self):
        return self.B
    
    def on_epoch_end(self):
        if self.shuffle:
            seed = np.random.randint(0, self.L)
            self.idx = np.concatenate([
                np.flip(np.arange(seed, 0, -self.seg_length)),
                np.arange(seed+self.seg_length, self.L, self.seg_length)
                ])
            self.idx = self.idx[:-1]
            random.shuffle(self.idx)

    def __getitem__(self, index):
       
        for i in range(self.batch_size):
            k = index*self.batch_size + i
            left = self.idx[k]
            right = left+self.seg_length
            self.spch[i] = self.speech[left:right]*1./(2 ** 15 - 1)
            exc = self.exct[left:right]*1./(2 ** 15)
            self.E[i] = frame_exct(exc, self.frame_length, self.frame_shift)
            self.X[i] = np.append(self.spch[i], np.zeros(176))
        M = mfcc(self.spch, frame_length=256, frame_step=80, fft_length=1024, num_mels=80, num_mfccs=60)
        return [M, self.E], [self.X, self.X, self.X, self.X] 



