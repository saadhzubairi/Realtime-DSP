# play_sine_02_blocks.py
# Play a sinusoid using Pyaudio.
# This program writes the signal in blocks.

from math import cos, pi 
import pyaudio, struct

RATE = 8000             # sampling rate (samples/second)
DURATION = 4            # duration (seconds)
N = DURATION * RATE     # number of samples
f1 = 200                # frequency of sinusoid (Hz)

# Create Pyaudio object
p = pyaudio.PyAudio()
stream = p.open(
    format = pyaudio.paInt16,  
    channels = 1, 
    rate = RATE,
    input = False, 
    output = True)

gain = 0.2 * 2**15
theta = 0
om1 = 2.0 * pi * f1 / RATE

# Play sin(2 * pi * f1 * n / RATE)

BLOCKLEN = 512
output_block = [0] * BLOCKLEN  # create 1D array

print('* Playing for %d seconds' % DURATION)

for n in range(0, int(N/BLOCKLEN)):
  for i in range(0, BLOCKLEN):
    output_block[i] = int( gain * cos(theta) )
    theta = theta + om1
  while theta > pi:
    theta = theta - 2.0 * pi
  binary_data = struct.pack('h' * BLOCKLEN, *output_block)   # 'h' for 16 bits
  stream.write(binary_data)

print('* Finished')

stream.stop_stream()
stream.close()
p.terminate()
