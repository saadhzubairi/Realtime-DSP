import struct
from matplotlib import pyplot
import math
import pyaudio

RATE = 8000
WIDTH = 2
CHANNELS = 1
DURATION = 20 #seconds
BLOCKLEN = 512
K = int(DURATION*RATE/BLOCKLEN) #number of blocks

def toMinute(seconds):
    mins = math.floor(seconds / 60)
    secs = int(seconds % 60)
    return f'{mins}:{secs:02d}'

pyplot.ion()
pyplot.figure(1)
plots = pyplot.plot([],[])
plot = plots[0]

#set x
n = range(0,BLOCKLEN)
pyplot.ylim(-10000,10000)
pyplot.xlim(0,BLOCKLEN)
pyplot.xlabel('Time (n)')
plot.set_xdata(n)

p = pyaudio.PyAudio()
stream = p.open(
    format=p.get_format_from_width(WIDTH),
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=False,
)


COUNT = 0


for i in range(K):
    
    input_bytes = stream.read(BLOCKLEN)
    #binary to tuple
    signal_block = struct.unpack('h'*BLOCKLEN, input_bytes)
    
    #set y data and pause
    plot.set_ydata(signal_block)
    plot.set_label(f'from time {toMinute(int((COUNT/RATE)*60))}s to time {toMinute(int(((COUNT + BLOCKLEN)/RATE)*60))}')
    COUNT += BLOCKLEN
    pyplot.legend()
    pyplot.pause(0.0001)    
    
stream.stop_stream()
stream.close()
p.terminate()
pyplot.ioff()
pyplot.show()


