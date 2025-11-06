import pyaudio, wave
import numpy as np
import numpy.typing as npt
import scipy.signal
import os
import math

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
ALPHA           = 1.2              # Scaling factor
OVERLAP_FACTOR  = 0.75
BLOCKLEN = 2048
MAXVALUE = 2**15-1  # Maximum allowed output signal value (because WIDTH = 2)

print('The file has %d channel(s).'            % CHANNELS)
print('The frame rate is %d frames/second.'    % RATE)
print('The file has %d frames.'                % signal_length)
print('There are %d bytes per sample.'         % WIDTH)

output_wf = wave.open(output_wavfile, 'w')      # wave file
output_wf.setframerate(RATE)
output_wf.setsampwidth(WIDTH)
output_wf.setnchannels(CHANNELS)

p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(
    format      = p.get_format_from_width(WIDTH),
    channels    = CHANNELS,
    rate        = RATE,
    input       = False,
    output      = True )


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
   
   
def frames_to_process_with_hops(all_frames, block_size,overlap_factor:int=0.5):
    frames = []
    if overlap_factor > 1 or overlap_factor < 0:
        print("[-] invalid overlap factor")
        return
    else:
        hop_count = math.floor(block_size*(1-overlap_factor))
        idx_i = 0
        idx_l = block_size
        while idx_l < len(all_frames)-1:
            print("from",idx_i,"to",idx_l)
            frames.append(all_frames[idx_i:idx_l])
            idx_i = idx_i + hop_count
            idx_l = idx_i + block_size
    return frames

binary_data = wf.readframes(signal_length)
all_samples = np.frombuffer(binary_data, dtype=np.int16)
frames = frames_to_process_with_hops(all_frames=all_samples, block_size=BLOCKLEN, overlap_factor=OVERLAP_FACTOR)
print("number of blocks:",len(frames))
# Get first set of frame from wave file0
for frame_idx in range(len(frames)):
    input_block = frames[frame_idx]
    
    if frame_idx == (len(frames)-1):
        next_input_block = []
    else:
        next_input_block = frames[frame_idx+1]
    
    # filter (here do the fft filtering thing)
    output_block = process_block_fft_scaling(input_block=input_block, alpha=ALPHA)
    if frame_idx == (len(frames)-1):
        next_output_block = []
    else:
        next_output_block = process_block_fft_scaling(input_block=next_input_block, alpha=ALPHA)
    
    hop_count = math.floor(BLOCKLEN*(1-OVERLAP_FACTOR))
    # averaging of the values of overlapping samples:
    for i in range(hop_count):
        if frame_idx == (len(frames)-1):
            continue
        else:
            output_block[hop_count+i] = (output_block[hop_count+i] + next_output_block[i]) * 0.5
    # clipping
    if frame_idx == 0:
        output_block = output_block
    else:
        output_block =output_block[hop_count:]
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
