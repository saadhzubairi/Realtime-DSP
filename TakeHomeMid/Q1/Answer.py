import pyaudio
import wave
import numpy as np
import os
import math
import tkinter as Tk

# function to make a preprocessed list of frames for overlapped block processing
def frames_to_process_with_hops(all_frames, block_size, overlap_factor):
    frames = []
    hop_count = math.floor(block_size * (1 - overlap_factor))
    idx_i = 0
    idx_l = block_size
    while idx_l < len(all_frames) - 1:
        frames.append(all_frames[idx_i:idx_l])
        idx_i += hop_count
        idx_l = idx_i + block_size
    return frames

# functiuon for processing blocks with scaling via fft and ifft with interpolation
def process_block_fft_scaling(input_block, alpha):
    X = np.fft.rfft(input_block)
    Y = np.zeros_like(X)
    for src_ind in range(X.size):
        dst_ind = src_ind / alpha
        if dst_ind < X.size - 1:
            i0 = int(np.floor(dst_ind))
            i1 = i0 + 1
            t = dst_ind - i0
            Y[src_ind] = (1 - t) * X[i0] + t * X[i1]
    y = np.fft.irfft(Y)
    return y



base_dir = os.path.dirname(os.path.abspath(__file__))
wavfile = os.path.join(base_dir, 'author.wav')
wf = wave.open(wavfile, 'rb')
#output_wavfile = 'author_output_blocks_corrected.wav'

#print('Play the wave file %s.' % wavfile)

# Open wave file (should be mono channel)
#wf = wave.open( wavfile, 'rb' )

CONTINUE = True # Variable for the looping mechanic
CHANNELS        =  wf.getnchannels()
RATE            = wf.getframerate()
WIDTH           = wf.getsampwidth()
signal_length   = wf.getnframes()
BLOCKLEN        = 1024
OVERLAP_FACTOR  = 0.5
MAXVALUE        = 2**15 - 1

print('The file has %d channel(s).'            % CHANNELS)
print('The frame rate is %d frames/second.'    % RATE)
print('The file has %d frames.'                % signal_length)
print('There are %d bytes per sample.'         % WIDTH)

#output_wf = wave.open(output_wavfile, 'w')      # wave file
#output_wf.setframerate(RATE)
#output_wf.setsampwidth(WIDTH)
#output_wf.setnchannels(CHANNELS)

Hop = int(BLOCKLEN * (1 - OVERLAP_FACTOR))
binary_data = wf.readframes(signal_length)
all_samples = np.frombuffer(binary_data, dtype=np.int16)

root = Tk.Tk()
root.title('Real-time Frequency Scaling')

# Scaling factor init
alpha = Tk.DoubleVar()
alpha.set(1.0)  
# print(alpha.get())

# Slider config here
alpha_slider = Tk.Scale(root, label='Scaling Factor (From 0.5 to 1)', variable=alpha, from_=0.5, to=2.0,resolution=0.01, orient=Tk.HORIZONTAL, length=300)
alpha_slider.pack(side=Tk.TOP)

# Quit button config here
def handle_close_quit():
    global CONTINUE
    CONTINUE = False
B_quit = Tk.Button(root, text='Quit', command=handle_close_quit)
B_quit.pack(side=Tk.BOTTOM, fill=Tk.X)

# Pyaudio config
p = pyaudio.PyAudio()
# Open audio stream
stream = p.open(
    format      = p.get_format_from_width(WIDTH),
    channels    = CHANNELS,
    rate        = RATE,
    input       = False,
    output      = True )


# Main loop
print('* Start')

prev_tail = np.zeros(BLOCKLEN - Hop)
frames = frames_to_process_with_hops(all_samples, BLOCKLEN, OVERLAP_FACTOR)
frame_idx = 0
while CONTINUE:
    root.update()
    
    # get slider value in real time
    alpha_from_slider = alpha.get()
    
    # get next input frame  
    input_block = frames[frame_idx]
    
    # process the scaled 
    output_block = process_block_fft_scaling(input_block, alpha_from_slider)

    # overlap and add
    output_block[:len(prev_tail)] = 0.5 * (output_block[:len(prev_tail)] + prev_tail)

    # clipping
    output_chunk = np.clip(output_block[:Hop], -MAXVALUE, MAXVALUE)
    output_chunk = np.around(output_chunk).astype(np.int16)
    stream.write(output_chunk.tobytes())
    
    prev_tail = output_block[Hop:]
    # increment to the next frame (circular)
    frame_idx = (frame_idx + 1) % len(frames)

print('* Finished')


stream.stop_stream()
stream.close()
p.terminate()
wf.close()


