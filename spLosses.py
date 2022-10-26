import tensorflow as tf
import numpy as np
from scipy import signal
from tensorflow.keras.losses import MeanSquaredError as mse
from tensorflow.python.framework.ops import disable_eager_execution


forward_window_fn=tf.signal.hann_window
def overlap_and_add(frames,frame_length=512, frame_step=256):
    #inv_window=1./forward_window_fn(512, dtype=tf.float32)
    inverse_stft_window_fn = tf.signal.inverse_stft_window_fn(frame_step=frame_step,
            forward_window_fn=forward_window_fn)
    inv_window = inverse_stft_window_fn(frame_length=frame_length, dtype=tf.float32)
    inv_window = tf.expand_dims(inv_window, axis=0)
    inv_window = tf.expand_dims(inv_window, axis=0)
    return tf.signal.overlap_and_add(frames*inv_window, frame_step=frame_step)


def magnitude_to_decibels(X):
    epsilon = 1e-8
    return 10*tf.math.log(tf.math.maximum(X, epsilon))/tf.math.log(10.)

def mel_spectrum(wav, frame_length=1024, frame_step=256, 
        fft_length=1024, num_mels=80):
    n_freq = fft_length//2
    mel_wt= tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mels, num_spectrogram_bins=n_freq+1, 
        sample_rate=16000,lower_edge_hertz=0., upper_edge_hertz=8000., 
        dtype=tf.dtypes.float32, name=None)
    
    spec = tf.signal.stft(wav,frame_length=frame_length, 
            frame_step=frame_step, fft_length=fft_length, pad_end=True)
    
    mag_spec = tf.math.abs(spec)
    mel_spec=tf.tensordot(mag_spec**2, mel_wt,1)
    mel_spec = magnitude_to_decibels(mel_spec)
    return mel_spec

def mr_stftloss1(y_true, y_pred):
    stft_loss1 = stftloss(y_true, y_pred, frame_length=600, frame_step=120, fft_length=1024)
    return stft_loss1

def mr_stftloss2(y_true, y_pred):
    stft_loss2 = stftloss(y_true, y_pred, frame_length=1200, frame_step=240, fft_length=2048)
    return stft_loss2

def mr_stftloss3(y_true, y_pred):
    stft_loss3 = stftloss(y_true, y_pred, frame_length=240, frame_step=50, fft_length=512)
    return stft_loss3

def stftloss(y_true, y_pred, frame_length, frame_step,  fft_length):
    S = tf.math.abs(tf.signal.stft(y_true, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length))
    N = tf.math.abs(tf.signal.stft(y_pred, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length))
    sc = tf.reduce_mean(tf.norm(S-N, ord='fro', axis=[-2,-1])/tf.norm(S, ord='fro', axis=[-2,-1]), axis=-1)
    mag = tf.reduce_mean(tf.math.abs(tf.math.log(S+1e-5)-tf.math.log(N+1e-5)), axis=-1)
    return 0.333*(sc+mag)

def mel_loss(y_true, y_pred):
    mel_true = mel_spectrum(y_true, frame_length=400, frame_step=80, fft_length=1024, num_mels=80)
    mel_pred = mel_spectrum(y_pred, frame_length=400, frame_step=80, fft_length=1024, num_mels=80)
    mel_loss = tf.reduce_mean(tf.math.abs(mel_true-mel_pred), axis=-1) 
    return mel_loss

