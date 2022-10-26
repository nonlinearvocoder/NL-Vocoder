import os
import numpy as np
from scipy.io import wavfile
from exc_from_f0 import synthesise_excitation

full_wav = []
excitation = []
outstem = 'tmp'
ana_comm = os.path.join(os.getcwd(), 'analysis')
x2x_comm = os.path.join(os.getcwd(), 'x2x')
data_path = os.path.join(os.getcwd(), 'data', 'wavs')

for wav_file in os.listdir(data_path):
    print(wav_file)
    wav_path = os.path.join(data_path, wav_file)
    fs, wav = wavfile.read(wav_path)
    wav_len = len(wav)
    comm = "%s %s %s.f0.double %s.sp.double %s.bap.double > %s.log"%(ana_comm, wav_path, outstem, outstem, outstem, outstem)
    success = os.system(comm)
    comm = "%s +da %s.bap.double > %s.bap"%(x2x_comm, outstem, outstem)
    success = os.system(comm)
    log_bap = np.loadtxt(outstem+".bap")
    bap = np.exp(log_bap)
    bap = np.repeat(bap, 80, axis=0)
    comm = "%s +da %s.f0.double > %s.txt"%(x2x_comm, outstem, outstem)
    success = os.system(comm)
    fo = np.loadtxt(outstem+".txt")
    exct = synthesise_excitation(fo, wav_len).astype(np.int16)
    bap = bap[:len(exct)] 
    exct = np.divide(exct/2**15 + np.multiply((np.random.rand(len(exct)))/2, bap) , 1.0+bap)
    exct *= 2**15
    exct = exct[:wav_len]
    excitation.append(exct.astype(np.int16))  
    full_wav.append(wav.astype(np.int16))

excitation = np.concatenate(excitation)
full_wav = np.concatenate(full_wav)
wavfile.write('excitation.wav', fs, excitation) 
wavfile.write('wave.wav', fs, full_wav)

for fil in ['tmp.bap', 'tmp.bap.double', 'tmp.f0.double', 'tmp.log', 'tmp.sp.double', 'tmp.txt']:
    os.remove(fil)
