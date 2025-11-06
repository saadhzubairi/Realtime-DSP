import pyaudio, wave
import numpy as np
import numpy.typing as npt
import scipy.signal
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
wavfile = os.path.join(base_dir, 'author.wav')
output_wavfile = os.path.join(base_dir, 'author_output_blocks_corrected.wav')

print('Play the wave file %s.' % wavfile)

# Open wave file (should be mono channel)
wf = wave.open( wavfile, 'rb' )

# Read the wave file properties
CHANNELS        = wf.getnchannels()     # Number of channels
RATE            = wf.getframerate()     # Sampling rate (frames/second)
signal_length   = wf.getnframes()       # Signal length
WIDTH           = wf.getsampwidth()     # Number of bytes per sample
ALPHA           = 0.5                   # Scaling factor

print('The file has %d channel(s).'            % CHANNELS)
print('The frame rate is %d frames/second.'    % RATE)
print('The file has %d frames.'                % signal_length)
print('There are %d bytes per sample.'         % WIDTH)

output_wf = wave.open(output_wavfile, 'w')      # wave file
output_wf.setframerate(RATE)
output_wf.setsampwidth(WIDTH)
output_wf.setnchannels(CHANNELS)

# Difference equation coefficients
b0 =  0.008442692929081
b2 = -0.016885385858161
b4 =  0.008442692929081
b = [b0, 0.0, b2, 0.0, b4]

# a0 =  1.000000000000000
a1 = -3.580673542760982
a2 =  4.942669993770672
a3 = -3.114402101627517
a4 =  0.757546944478829
a = [1.0, a1, a2, a3, a4]

p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(
    format      = p.get_format_from_width(WIDTH),
    channels    = CHANNELS,
    rate        = RATE,
    input       = False,
    output      = True )

BLOCKLEN = 1024
MAXVALUE = 2**15-1  # Maximum allowed output signal value (because WIDTH = 2)

# Get first set of frame from wave file0
binary_data = wf.readframes(BLOCKLEN)

# ORDER = 4   # filter is fourth order
# states = np.zeros(ORDER)

def process_block_fft_scaling(input_block:npt.NDArray,alpha:float):    
    X = np.fft.rfft(input_block)
    Y = np.zeros_like(X)
    # here's where we shift or rather scale by alpha:
    for i in range(X.size):
        src_idx = i/alpha
        if(src_idx < X.size - 1):
            #interpolating between the bins
            i_0 = int(np.floor(src_idx))
            i_1 = i_0 + 1
            t = src_idx - i_0
            Y[i] = (1-t)*X[i_0] + (t)*X[i_1]            
        # else it'll be 0
    resultant = np.fft.irfft(Y)
    return resultant
    

while len(binary_data) == WIDTH * BLOCKLEN:

    # convert binary data to numbers
    # input_block = struct.unpack('h' * BLOCKLEN, binary_data) 
    input_block = np.frombuffer(binary_data, dtype = 'int16') # use Numpy

    # filter (here do the fft filtering thing)
    output_block = process_block_fft_scaling(input_block=input_block, alpha=ALPHA)

    # clipping
    output_block = np.clip(output_block, -MAXVALUE, MAXVALUE)

    # convert to integer
    output_block = np.around(output_block)          # round to integer
    output_block = output_block.astype('int16')     # convert to 16-bit integer

    # Convert output value to binary data
    # binary_data = struct.pack('h' * BLOCKLEN, *output_block)
    binary_data = output_block.tobytes() # using Numpy

    # Write binary data to audio stream
    stream.write(binary_data)

    # Write binary data to output wave file
    output_wf.writeframes(binary_data)

    # Get next frame from wave file
    binary_data = wf.readframes(BLOCKLEN)

print('* Finished')

stream.stop_stream()
stream.close()
p.terminate()

# Close wavefiles
wf.close()
output_wf.close()
