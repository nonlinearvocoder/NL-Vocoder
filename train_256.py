import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, History, ReduceLROnPlateau
import tensorflow_addons as tfa
import h5py, os
import numpy as np
from functools import partial, update_wrapper
from spLosses import mr_stftloss1, mr_stftloss2, mr_stftloss3, overlap_and_add, mel_loss
from scipy import signal
from custom import dg_nlvocoder
from swan_new import MHAttn

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

def swan(inputs, n_blocks=1, n_heads=4, head_size=64, context=10, out_dim=256):
    x = tf.keras.layers.Dense(out_dim, activation='relu')(inputs)
    for i in range(n_blocks):
        cx =  MHAttn(n_heads, head_size, context)(x)
        x = tf.keras.layers.BatchNormalization()(x+cx)
        xe = tf.keras.layers.Dense(out_dim, activation='relu')(x)
        xe = tf.keras.layers.Dense(out_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)    
    
    x = tf.keras.layers.Dense(out_dim, activation='relu')(x)
    return x

dg_train = dg_nlvocoder('wave.wav', 'excitation.wav', batch_size = 16, frame_length=256)

regularizer = tf.keras.regularizers.L2(0)
initializer = tf.keras.initializers.GlorotUniform() 

if os.path.exists('mfcc_mean_256.npy'):
    mean = np.load('mfcc_mean_256.npy')
    variance = np.load('mfcc_var_256.npy')
else:
    #Get Normalization
    x=[]
    for i in range(400):
        [M,_], _ = dg_train.__getitem__(i)
        x.append(M)   #.numpy()

    X = np.vstack(x)
    mean = np.mean(X, axis=(0,1))
    variance = np.var(X, axis=(0,1))
    np.save('mfcc_mean_256.npy', mean)
    np.save('mfcc_var_256.npy', variance)

Normalizer = tf.keras.layers.experimental.preprocessing.Normalization(
        axis=-1, mean=mean, variance=variance)

mfcc = tf.keras.Input(shape=(None,60), dtype=tf.float32)
exct = tf.keras.Input(shape=(None,256), dtype=tf.float32)
norm_mfcc = Normalizer(mfcc)
mfc = tf.keras.layers.Dense(128, activation='linear')(norm_mfcc)
filtr = swan(mfc, n_blocks=4, n_heads=4, head_size=64, context=12, out_dim=256)

exc_spec = tf.keras.layers.Lambda(lambda z: tf.signal.fft(tf.cast(z, tf.complex64)))(exct)
prod = tf.keras.layers.Multiply()([exc_spec, tf.cast(filtr, tf.complex64)])
convolved = tf.keras.layers.Lambda(lambda z: tf.signal.ifft(z))(prod)
convolved = tf.keras.layers.Lambda(lambda z: tf.math.real(z))(convolved)

est_wav = tf.keras.layers.Lambda(lambda z: overlap_and_add(z, frame_length=256, frame_step=80))(convolved)
est_wav = tf.expand_dims(est_wav, axis=-1)
wav = tf.keras.layers.Conv1D(1, 3, padding='same',activation='linear', name='wav_mel', kernel_regularizer=regularizer)(est_wav)
mr1 = tf.keras.layers.Lambda(lambda z: z, name='mr1')(wav)
mr2 = tf.keras.layers.Lambda(lambda z: z, name='mr2')(wav)
mr3 = tf.keras.layers.Lambda(lambda z: z, name='mr3')(wav)

model= tf.keras.models.Model(inputs=[mfcc, exct], outputs=[wav, mr1, mr2, mr3])
model.summary()

lr=1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    lr, decay_steps=5000, decay_rate=0.95, staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model_check = ModelCheckpoint('weights_256/weights-{epoch:04d}.h5', monitor='loss', save_best_only=True)
early_stop=EarlyStopping(monitor='loss', patience=50)
model.compile(optimizer=optimizer,
        loss={'mr1':mr_stftloss1, 'mr2':mr_stftloss2, 'mr3':mr_stftloss3, 'wav_mel':mel_loss}, 
        loss_weights={'mr1': 1, 'mr2': 1, 'mr3': 1, 'wav_mel':0.1})  
model.fit(dg_train, epochs=300, verbose=1, 
        callbacks=[model_check, early_stop], 
        max_queue_size=10, workers=1, use_multiprocessing=False,
        )
