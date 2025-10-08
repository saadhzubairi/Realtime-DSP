import struct
import wave 
from matplotlib import pyplot
import math
import pyaudio

f0 = 0 #(hz)

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

p = pyaudio.PyAudio()
stream = p.open(
    format=p.get_format_from_width(WIDTH),
    channels=CHANNELS,
    rate=RATE,
    input=False,
    output=True,
)

for n in range(0,LEN):
    input_bytes = wf.readframes(1)
    
    input_tuple = struct.unpack('h',input_bytes)
    
    x = input_tuple[0]
    y = x * math.cos(2.0 * math.pi * f0 * n/RATE)
    
    output_bytes = struct.pack('h',int(y))
    
    stream.write(output_bytes)

print('finished')

stream.stop_stream()    
stream.close()
p.terminate()