from struct import pack
from math import sin, pi
import wave

Fs = 8000

wf = wave.open('sin_01_mono.wav','w')
wf.setnchannels(1)      # one channel
wf.setsampwidth(2)      # two [bytes] per sample
wf.setframerate(Fs)     # samples per second
# the framerate in this context means the number of samples per second times channels (8000 for stereo audio would be 16000 samples per second)

Amplitude = 2*15 - 1.0   # amp
frequency = 19791.0       # hertz (note a3 on the piano)
N = int(5*Fs)         # one and a half second in samples

for n in range (0, N):
    x = Amplitude * sin(2*pi*frequency*n/Fs)    #signal value
    byte_string = pack('H', int(x)) # h is for short integer (16 bits)
    wf.writeframes(byte_string)
wf.close()

