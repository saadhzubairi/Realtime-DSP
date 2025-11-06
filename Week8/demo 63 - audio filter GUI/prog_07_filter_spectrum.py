# plot_play_file_filter_spectrum_numpy.py

import pyaudio
import wave
import numpy as np
from scipy.signal import butter, lfilter, freqz
import tkinter as Tk
from matplotlib import animation
import matplotlib.figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')      # Either works...

# BLOCKLEN = 512
BLOCKLEN = 256          # no drop outs
# BLOCKLEN = 128      # DROP OUTS
# BLOCKLEN = 64       # DROP OUTS

# Specify wave file
wavfile = 'author.wav'
print('Name of wave file: %s' % wavfile)

# Open wave file
wf = wave.open( wavfile, 'rb')

# Read wave file properties
RATE        = wf.getframerate()     # Frame rate (frames/second)
WIDTH       = wf.getsampwidth()     # Number of bytes per sample
LEN         = wf.getnframes()       # Signal length
CHANNELS    = wf.getnchannels()     # Number of channels

print('The file has %d channel(s).'         % CHANNELS)
print('The file has %d frames/second.'      % RATE)
print('The file has %d frames.'             % LEN)
print('The file has %d bytes per sample.'   % WIDTH)

BLOCK_DURATION = 1000.0 * BLOCKLEN/RATE # duration in milliseconds
print('Block length: %d' % BLOCKLEN)
print('Duration of block in milliseconds: %.2f' % BLOCK_DURATION)


# Update filter when slider is moved
def update_filter(event):
    global a, b
    global fc
    fc = cutoff_freq.get()
    # [b, a] = butter(ORDER, 2 * fc)
    [b, a] = butter(ORDER, 2 * fc, btype = 'high')
    [om, H] = freqz(b, a)
    
    line_freqresp.set_ydata(np.abs(H))
    # fig1.canvas.draw()


# DEFINE TKINTER ROOT

root = Tk.Tk()    # Define root before creating figure

# Cut-off frequency slider
cutoff_freq = Tk.DoubleVar()            # Define Tk variable
cutoff_freq.set(0.25)                   # Initilize
S_cutoff = Tk.Scale(root, label = 'Cut-off frequency',
        variable = cutoff_freq, from_ = 0.01, to = 0.49, resolution = 0.01,
        orient = 'horizontal',
        command = update_filter)

# Quit button
B_quit = Tk.Button(root, text = 'Quit', command = root.quit)


# Create Butterworth filter
# fc = cutoff_freq.get()
# [b, a] = butter(ORDER, 2 * fc, btype = 'high')

# Initialize filter
b = [0]
a = [1]
[om, H] = freqz(b, a)
f = om/(2 * np.pi)
ORDER = 2   # filter order
states = np.zeros(ORDER)

# DEFINE FIGURE

fig1 = matplotlib.figure.Figure()                     # not using pyplot
ax_freqresp = fig1.add_subplot(3, 1, 1)
ax_signal = fig1.add_subplot(3, 1, 2)
ax_spectrum = fig1.add_subplot(3, 1, 3)
fig1.set_size_inches((6, 8))  # (width, height)

n = np.arange(BLOCKLEN)
output_block = np.zeros(BLOCKLEN)
[line_signal] = ax_signal.plot(n, output_block)

ax_signal.set_ylim(-5000, 5000)
ax_signal.set_xlim(0, BLOCKLEN)
ax_signal.set_xlabel('Time (n)')
ax_signal.set_title('Output signal')

[line_freqresp] = ax_freqresp.plot(f, np.abs(H))
ax_freqresp.set_xlim(0, 0.5)
ax_freqresp.set_ylim(0, 1.2)
ax_freqresp.set_title('Frequency response')
ax_freqresp.set_xlabel('Frequency (cycles/sample)')

X = np.fft.rfft(output_block)  # real fft
f2 = np.arange(X.size) / BLOCKLEN
[line_spectrum] = ax_spectrum.plot(f2, np.abs(X)/BLOCKLEN)
ax_spectrum.set_xlim(0, 0.5)
ax_spectrum.set_ylim(0, 100)
ax_spectrum.set_title('Spectrum of output signal')
ax_spectrum.set_xlabel('Frequency (cycles/sample)')

fig1.tight_layout()


# Turn figure into a Tkinter widget
my_canvas = FigureCanvasTkAgg(fig1, master = root)
# fig1.canvas.draw()   # (optional ?)
# my_canvas.draw()   # (optional ?)
C1 = my_canvas.get_tk_widget()   # canvas widget

# PLACE WIDGETS

C1.pack()
B_quit.pack(side = 'right', expand = True, fill = 'both')
S_cutoff.pack(side = 'left', expand = True, fill = 'both')

# Open the audio output stream
p = pyaudio.PyAudio()

PA_FORMAT = p.get_format_from_width(WIDTH)
stream = p.open(
    format = PA_FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = False,
    output = True,
    frames_per_buffer = BLOCKLEN)

output_block = np.zeros(BLOCKLEN)
MAXVALUE = 2**15-1

def my_init():
    print('hello')
    return (line_signal, line_freqresp, line_spectrum)
    
def my_update(i):

    global states

    # Get block of samples from audion input stream
    input_bytes = wf.readframes(BLOCKLEN)
    # input_bytes = stream.read(BLOCKLEN, exception_on_overflow = False)

    # Rewind if at end of file
    if len(input_bytes) < WIDTH * BLOCKLEN:
        wf.rewind()
        input_bytes = wf.readframes(BLOCKLEN)

    # Convert from binary data
    input_block = np.frombuffer(input_bytes, dtype = 'int16')

    # filtering
    [output_block, states] = lfilter(b, a, input_block, zi = states)

    # clip before converting to 16 bit integers
    signal_block = np.clip(output_block, -MAXVALUE, MAXVALUE)   
        
    line_signal.set_ydata(output_block)
    line_spectrum.set_ydata(np.abs(np.fft.rfft(output_block))/BLOCKLEN)

    # Convert output value to binary data
    # And write binary data to audio output stream
    stream.write(output_block.astype('int16').tobytes(), BLOCKLEN)

    return (line_signal, line_freqresp, line_spectrum)

update_filter(None)         # draw frequency response

my_anima = animation.FuncAnimation(
    fig1,
    my_update,
    init_func = my_init,
    interval = 10,   # milliseconds (what happens if this is 200?)
    blit = True,
    cache_frame_data = False,
    repeat = False
)

Tk.mainloop()           # Start Tkinter (includes animation)

stream.stop_stream()
stream.close()
p.terminate()

print('* Finished')
