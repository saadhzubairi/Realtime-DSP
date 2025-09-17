from struct import pack
from math import sin, pi
import wave

Fs = 8000

wf = wave.open('sin_01_stereo.wav','w')
wf.setnchannels(2)      # one channel
wf.setsampwidth(2)      # two [bytes] per sample
wf.setframerate(Fs)     # samples per second
# the framerate in this context means the number of samples per second times channels (8000 for stereo audio would be 16000 samples per second)

Amplitude = 2*15 - 1.0   # amp
frequency1 = 19791.0       # hertz (note a3 on the piano)
frequency2 = 220.0        # hertz (note a3 on the piano)
N = int(5*Fs)         # one and a half second in samples

for n in range (0, N):
    # left channel
    x = Amplitude * sin(2*pi*frequency1*n/Fs)    #signal value
    byte_string = pack('h', int(x)) # h is for short integer (16 bits)
    wf.writeframes(byte_string)
    
    # right channel
    x = Amplitude * sin(2*pi*frequency2*n/Fs)    #signal value
    byte_string = pack('h', int(x)) # h is for short integer (16 bits)
    wf.writeframes(byte_string)
    
wf.close()

