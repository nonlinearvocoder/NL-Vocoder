import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import tensorflow.keras.backend as K
import tensorflow as tf
from functools import partial, update_wrapper
from scipy.io import wavfile
from spLosses import * 
from scipy import signal
from custom import frame_exct, mfcc
from swan_new import MHAttn
from exc_from_f0 import synthesise_excitation
import timeit
import argparse


def analysis(input_file, wav, frame_length=256, frame_step=80, fft_length=1024):
    mfccs = mfcc(wav, frame_length, frame_step, fft_length, num_mels=80, num_mfccs=60)
    comm = "%s %s %s.f0.double %s.sp.double %s.bap.double > %s.log"%(ana_comm, input_file, outstem, outstem, outstem, outstem)
    success = os.system(comm)
    comm = "%s +da tmp.bap.double > tmp.bap"%(x2x_comm)
    success = os.system(comm)
    log_bap = np.loadtxt("tmp.bap")
    bap = np.exp(log_bap)
    comm = "%s +da tmp.f0.double > tmp.txt"%(x2x_comm)
    success = os.system(comm)
    fo = np.loadtxt("tmp.txt")
    return fo, bap, mfccs

def generate_exc(fo, bap, wavlen):
    sawtooth = synthesise_excitation(fo, wavlen)
    bap = np.repeat(bap, 80, axis=0)
    bap = bap[:len(sawtooth)]
    new_exct = np.divide(sawtooth/2**15 + np.multiply((np.random.rand(len(sawtooth)))/2, bap) , 1.0+bap)
    return new_exct

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to saved model')
parser.add_argument('--input', required=True, help='Path to test wavfile')
parser.add_argument('--pitch_factor', default=1.0, help='Pitch modification factor')
args = parser.parse_args()

input_file = args.input
file_name = args.input.split('/')[-1].split('.')[0]
epoch = int(args.checkpoint.split('-')[-1].split('.')[0])
fo_mod_factor = float(args.pitch_factor)
outstem = 'tmp'
ana_comm = os.path.join(os.getcwd(), 'analysis')
x2x_comm = os.path.join(os.getcwd(), 'x2x')
frame_length=256
frame_step=80

model=load_model(args.checkpoint,custom_objects={
    'mfcc': mfcc, 
    'mr_stftloss1': mr_stftloss1,
    'mr_stftloss2': mr_stftloss2,
    'mr_stftloss3': mr_stftloss3,
    'MHAttn': MHAttn,
    'mel_loss':mel_loss,
    })
#model.summary()


sr, wav = wavfile.read(input_file)
wav = (wav*1./(2 ** 15 - 1)).astype(np.float32)
fo, bap, mfcc = analysis(input_file, wav, frame_length, frame_step)

start_time = timeit.default_timer()
duration = float(len(wav)/sr)
fo = fo*fo_mod_factor
exc = generate_exc(fo, bap, len(wav))
exc = exc.astype(np.float32)
exc = frame_exct(exc, frame_length, frame_step)
exc = np.expand_dims(exc, axis=-1).astype(np.float32)
exc = np.expand_dims(exc, axis=0).astype(np.float32)
mfcc = np.expand_dims(mfcc, axis=0)

est_wav,_,_,_ = model.predict([mfcc, exc])
#print ('Synthesis time: %.2f seconds' % (timeit.default_timer() - start_time) )
#print ('Real-time factor: %.2f ' % ((timeit.default_timer() - start_time) /duration))
est_wav = tf.reshape(est_wav, [-1]).numpy()* (2 ** 15 - 1)
wavfile.write(os.path.join('synthesized_256', file_name+'_'+str(epoch)+'_resyn.wav'), 16000, est_wav.astype(np.int16))

for fil in ['tmp.bap', 'tmp.bap.double', 'tmp.f0.double', 'tmp.log', 'tmp.sp.double', 'tmp.txt']:
    os.remove(fil)
