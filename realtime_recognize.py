# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import librosa
from model import *
import data
import time
from pyaudio import PyAudio, paInt16, paFloat32
import wave
import sys
import math

__author__ = 'doyleliu96@gmail.com'

CHUNK = 1024
SHORT_NORMALIZE = (1.0/32768.0)
global prev_mfcc
global mfcc
global flag


class recoder:
    NUM_SAMPLES = 1024      # pyaudio内置缓冲大小
    SAMPLING_RATE = 16000    # 取样频率
    LEVEL = 500         # 声音保存的阈值
    COUNT_NUM = 20      # NUM_SAMPLES个取样之内出现COUNT_NUM个大于LEVEL的取样则记录声音
    SAVE_LENGTH = 8         # 声音记录的最小长度：SAVE_LENGTH * NUM_SAMPLES 个取样
    TIME_COUNT = 60     # 录音时间，单位s
    

    Voice_String = []

    def savewav(self, filename):
        wf = wave.open(filename, 'wb') 
        wf.setnchannels(1) 
        wf.setsampwidth(2) 
        wf.setframerate(self.SAMPLING_RATE) 
        wf.writeframes(np.array(self.Voice_String).tostring()) 
        # wf.writeframes(self.Voice_String.decode())
        wf.close() 

    def recoder(self):
        pa = PyAudio() 
        stream = pa.open(format= paInt16, channels=1, rate=self.SAMPLING_RATE, input=True, 
            frames_per_buffer=self.NUM_SAMPLES) 
        save_count = 0 
        save_buffer = []
        wav_buffer = [] 
        signal_cum = []
        time_count = self.TIME_COUNT
        wav_count = 0

        while True:
            time_count -= 1
            # print time_count
            # 读入NUM_SAMPLES个取样
            string_audio_data = stream.read(self.NUM_SAMPLES, exception_on_overflow=False)
            # print(string_audio_data)
            # amplitude = self.get_rms(string_audio_data)
            # print(amplitude)

            
            # 将读入的数据转换为数组
            audio_data = np.fromstring(string_audio_data, dtype=np.short)
            wav_data = np.fromstring(string_audio_data, dtype=np.int16)

            # for each in wav_data:
            #     print(each)
            # 计算大于LEVEL的取样的个数
            large_sample_count = np.sum( audio_data > self.LEVEL )
            # print(np.max(audio_data))
            # 如果个数大于COUNT_NUM，则至少保存SAVE_LENGTH个块
            if large_sample_count > self.COUNT_NUM:
                save_count = self.SAVE_LENGTH 
            else: 
                save_count -= 1

            if save_count < 0:
                save_count = 0 

            if save_count > 0 : 
            # 将要保存的数据存放到save_buffer中
                #print  save_count > 0 and time_count >0
                save_buffer.append(string_audio_data)
                # wav_buffer.extend(wav_data)

                self.Voice_String = string_audio_data
                # wav_name = "test/%d.wav" % wav_count
                # wav_name = "test/tmp.wav"
                # self.savewav(wav_name)
                # wav_count = wav_count + 1
                wav_numpy = np.hstack(wav_data)
                wav_numpy = wav_numpy / (2.0 ** 15)

                # wav, _ = librosa.load(wav_name, mono=True, sr=16000)
                wav = wav_numpy
                global prev_mfcc
                global flag
                global mfcc
                if(flag is False):
                    prev_mfcc = librosa.feature.mfcc(wav, 16000)
                    mfcc = np.transpose(np.expand_dims(prev_mfcc, axis=0), [0, 2, 1])
                    flag = True
                else:
                    prev_mfcc = np.column_stack((prev_mfcc, librosa.feature.mfcc(wav, 16000)))
                    # mfcc = np.column_stack((mfcc, np.transpose(np.expand_dims(prev_mfcc, axis=0), [0, 2, 1])))
                    mfcc = np.transpose(np.expand_dims(prev_mfcc, axis=0), [0, 2, 1]) 


            else: 
            #print save_buffer
            # 将save_buffer中的数据写入WAV文件，WAV文件的文件名是保存的时刻
                
                if len(save_buffer) > 0 : 
                    self.Voice_String = save_buffer
                    # self.Voice_String = wav_buffer
                    for i,num in enumerate(wav_buffer):
                        if(math.isnan(num)):
                            wav_buffer[i] = 0
                    # print(len(wav_buffer))
                    # wav_numpy = np.hstack(wav_buffer)
                    # np.set_printoptions(threshold='nan') 
                   
                    save_buffer = [] 
                    wav_buffer = []
                    # global mfcc
                    # global flag
                    # flag = False
                    print("Recode a piece of voice successfully!")
                    # self.savewav("test/test.wav")
                    return True
            if time_count==0: 
                if len(save_buffer)>0:
                    self.Voice_String = save_buffer
                    save_buffer = [] 
                    wav_buffer = []
                    # wav_name = "test/%d.wav" % wav_count
                    # self.savewav(wav_name)
                    # wav_count = wav_count + 1
                    flag = False
                    print("Recode a piece of  voice successfully!")
                    # self.savewav("test/test.wav")
                    return True
                else:
                    return False

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 1     # batch size

#
# inputs
#

# vocabulary size
voca_size = data.voca_size

# mfcc feature of audio
x = tf.placeholder(dtype=tf.sg_floatx, shape=(batch_size, None, 20))

# sequence length except zero-padding
seq_len = tf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)

# encode audio feature
logit = get_logit(x, voca_size=voca_size)

# ctc decoding
decoded, _ = tf.nn.ctc_beam_search_decoder(logit.sg_transpose(perm=[1, 0, 2]), seq_len, merge_repeated=False)

# to dense tensor
y = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1

#
# regcognize wave file
#

# command line argument for input wave file path
tf.sg_arg_def(file=('', 'speech wave file to recognize.'))

# # load wave file
# wav, _ = librosa.load(tf.sg_arg().file, mono=True, sr=16000)
# # get mfcc feature
# mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, 16000), axis=0), [0, 2, 1])

# run network
with tf.Session() as sess:

    # init variables
    tf.sg_init(sess)
    # mfcc = np.zeros(20,)
    flag = False
    r = recoder()

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train'))
    # run session
    # t1 = time.time()
    print("record begin!")
    while(True):
        r.recoder()
        global mfcc
        label = sess.run(y, feed_dict={x: mfcc})
        # print(" ")
        data.print_index(label)
    # t2 = time.time()

    # print label
    
    # print(t2-t1)
