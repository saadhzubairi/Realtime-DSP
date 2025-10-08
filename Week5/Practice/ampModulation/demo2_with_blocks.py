import struct
import wave 
from matplotlib import pyplot
import math
import pyaudio
import os
wavefile = os.path.join(os.path.dirname(__file__), 'author.wav')
wf = wave.open(wavefile, 'rb')
f0 = 400 #(hz)

RATE = wf.getframerate()
WIDTH = wf.getsampwidth()
LEN = wf.getnframes()
CHANNELS = wf.getnchannels()

print(RATE)
print(WIDTH)
print(LEN)
print(CHANNELS)

BLOCKLEN = 64

p = pyaudio.PyAudio()
stream = p.open(
    format=p.get_format_from_width(WIDTH),
    channels=CHANNELS,
    rate=RATE,
    input=False,
    output=True,
)

output_block = BLOCKLEN*[0]

num_blocks = int(math.floor(LEN/BLOCKLEN))

for i in range(0,num_blocks):
    input_bytes = wf.readframes(BLOCKLEN)
    
    input_tuple = struct.unpack('h'*BLOCKLEN,input_bytes)
    
    for n in range(0,BLOCKLEN):
        #modulating every single value
        output_block[n] = int(input_tuple[n] * math.cos(2*math.pi*n*f0/RATE))
    
    output_bytes = struct.pack('h'*BLOCKLEN,*output_block)
    
    stream.write(output_bytes)

print('finished')

stream.stop_stream()    
stream.close()
p.terminate()