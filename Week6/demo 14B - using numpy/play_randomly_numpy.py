# play_randomly_numpy.py
"""
PyAudio Example: Generate random pulses and input them to an IIR filter of 2nd order.
"""

import pyaudio
import numpy as np
# import struct         # not needed - use numpy
# from math import sin, cos, pi    # not needed - use numpy

BLOCKSIZE = 1024    # Blocksize
WIDTH = 2           # Bytes per sample
CHANNELS = 1
RATE = 8000        # Sampling Rate in Hz

# Parameters
T = 10      # Total play time (seconds)
Ta = 0.8    # Decay time (seconds)
f1 = 350    # Frequency (Hz)

# Pole radius and angle
r = 0.01**(1.0/(Ta*RATE))       # 0.01 for 1 percent amplitude
om1 = 2.0 * np.pi * float(f1)/RATE

# Filter coefficients (second-order IIR)
a1 = -2 * r * np.cos(om1)
a2 = r**2
b0 = np.sin(om1)

NumBlocks = int( T * RATE / BLOCKSIZE )

y_block = np.zeros(BLOCKSIZE)

# Open the audio output stream
p = pyaudio.PyAudio()
PA_FORMAT = p.get_format_from_width(WIDTH)
stream = p.open(format = PA_FORMAT,
                channels = CHANNELS,
                rate = RATE,
                input = False,
                output = True)

print('Playing for {0:f} seconds ...'.format(T))

THRESHOLD = 2.5 / RATE

# Loop through blocks
for i in range(0, NumBlocks):

    # Do difference equation for block
    for n in range(BLOCKSIZE):

        rand_val = np.random.rand()
        if rand_val < THRESHOLD:
            x = 15000
        else:
            x = 0

        y_block[n] = b0 * x - a1 * y_block[n-1] - a2 * y_block[n-2]  
              # What happens when n = 0?
              # In Python negative indices cycle to end, so it works..

    output_block = np.clip(y_block, -32000, 32000)        # clipping
    output_block = np.around(output_block)          # round to integer
    output_block = output_block.astype('int16')     # convert to 16-bit integer
    binary_data = output_block.tobytes()            # convert to binary

    # Write binary string to audio output stream
    stream.write(binary_data, BLOCKSIZE)

print('* Finished')

stream.stop_stream()
stream.close()
p.terminate()
