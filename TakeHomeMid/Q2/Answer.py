import pyaudio
import wave
import numpy as np
import os
import math
import tkinter as Tk
import matplotlib
from matplotlib import pyplot
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use('TkAgg')

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
BLOCKLEN        = 256
OVERLAP_FACTOR  = 0.5
MAXVALUE        = 2**15 - 1

BLOCK_DURATION = 1000.0 * BLOCKLEN / RATE
print('Block length: %d' % BLOCKLEN)
print('Duration of block in milliseconds: %.2f' % BLOCK_DURATION)

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
alpha_slider = Tk.Scale(root, label='Scaling Factor (From 0.5 to 2)', variable=alpha, from_=0.5, to=2.0,resolution=0.01, orient=Tk.HORIZONTAL, length=300)
#alpha_slider.pack(side=Tk.TOP)

# Quit button config here
def handle_close_quit():
    root.destroy()
root.protocol("WM_DELETE_WINDOW", handle_close_quit)
B_quit = Tk.Button(root, text='Quit', command=handle_close_quit)

# DEFINE FIGURE
fig1 = matplotlib.figure.Figure()                     # not using pyplot
ax_inpFFT = fig1.add_subplot(2, 1, 1)
ax_outFFT = fig1.add_subplot(2, 1, 2)
fig1.set_size_inches((6, 8))  # (width, height)

n = np.arange(BLOCKLEN)

output_block = np.zeros(BLOCKLEN)
input_block = np.zeros(BLOCKLEN)
y_lim = 1750
X_in = np.fft.rfft(input_block)  # real fft
f1 = np.arange(X_in.size) / BLOCKLEN
[line_org_spectra] = ax_inpFFT.plot(f1, np.abs(X_in)/BLOCKLEN)
ax_inpFFT.set_xlim(0, 0.5)
ax_inpFFT.set_ylim(0, y_lim)
ax_inpFFT.set_title('Spectrum of input signal')
ax_inpFFT.set_xlabel('Frequency (cycles/sample)')

X_out = np.fft.rfft(output_block)  # real fft
f2 = np.arange(X_out.size) / BLOCKLEN
[line_con_spectra] = ax_outFFT.plot(f2, np.abs(X_out)/BLOCKLEN)
ax_outFFT.set_xlim(0, 0.5)
ax_outFFT.set_ylim(0, y_lim)
ax_outFFT.set_title('Spectrum of output signal')
ax_outFFT.set_xlabel('Frequency (cycles/sample)')

fig1.tight_layout()

# Turn figure into a Tkinter widget
my_canvas = FigureCanvasTkAgg(fig1, master = root)
# fig1.canvas.draw()   # (optional ?)
# my_canvas.draw()   # (optional ?)
C1 = my_canvas.get_tk_widget()   # canvas widget

# PLACE WIDGETS

C1.pack()
B_quit.pack(side = 'right', expand = True, fill = 'both')
alpha_slider.pack(side = 'left', expand = True, fill = 'both')

# Pyaudio config
p = pyaudio.PyAudio()
# Open audio stream
stream = p.open(
    format      = p.get_format_from_width(WIDTH),
    channels    = CHANNELS,
    rate        = RATE,
    input       = False,
    output      = True,
    frames_per_buffer = BLOCKLEN
    )

# Main loop
print('* Start')

def my_init():
    print('hello')
    return (line_org_spectra, line_con_spectra)

prev_tail = np.zeros(BLOCKLEN - Hop)
frames = frames_to_process_with_hops(all_samples, BLOCKLEN, OVERLAP_FACTOR)
frame_idx = 0

def my_update(i):
    global frame_idx
    global prev_tail
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
    
    # write to the graph variables
    line_org_spectra.set_ydata(np.abs(np.fft.rfft(input_block))/BLOCKLEN)
    line_con_spectra.set_ydata(np.abs(np.fft.rfft(output_block))/BLOCKLEN)

    stream.write(output_chunk.tobytes())
    
    prev_tail = output_block[Hop:]
    # increment to the next frame (circular)
    frame_idx = (frame_idx + 1) % len(frames)
    
    return (line_org_spectra, line_con_spectra)

print('* Finished')


my_anima = animation.FuncAnimation(
    fig1,
    my_update,
    init_func = my_init,
    interval = 10,   # milliseconds (what happens if this is 200?)
    blit = True,
    cache_frame_data = False,
    repeat = False
)

Tk.mainloop()

stream.stop_stream()
stream.close()
p.terminate()
wf.close()


