import struct
import wave 
from matplotlib import pyplot
import math

wavefile = 'author.wav'
wf = wave.open(wavefile, 'rb')

RATE = wf.getframerate()
WIDTH = wf.getsampwidth()
LEN = wf.getnframes()
CHANNELS = wf.getnchannels()

print(RATE)
print(WIDTH)
print(LEN)
print(CHANNELS)

BLOCKLEN = 250


def toMinute(seconds):
    mins = math.floor(seconds / 60)
    secs = int(seconds % 60)
    return f'{mins}:{secs:02d}'

pyplot.ion()

plots = pyplot.plot([],[])
plots[0].set_xdata(range(BLOCKLEN))


pyplot.ylim(-32000,32000)
pyplot.xlim(0,BLOCKLEN)

input_bytes = wf.readframes(BLOCKLEN)

COUNT = 0



while len(input_bytes) >= BLOCKLEN*WIDTH:
    
    #binary to tuple
    signal_block = struct.unpack('h'*BLOCKLEN, input_bytes)
    
    #set y data and pause
    plots[0].set_ydata(signal_block)
    plots[0].set_label(f'from time {toMinute(int((COUNT/RATE)*60))}s to time {toMinute(int(((COUNT + BLOCKLEN)/RATE)*60))}')
    COUNT += BLOCKLEN
    pyplot.legend()
    pyplot.pause(0.05)    
    input_bytes = wf.readframes(BLOCKLEN)
    
    
wf.close()
pyplot.ioff()
pyplot.show()


