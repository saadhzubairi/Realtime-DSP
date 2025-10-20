# filtering_toggle.py

# Filtering - plot input and output signals, plot frequency response.
# Use slider to adjust filter cut-off frequency.

# Input is from a wave file with looping

# Toggle between LPF (lowpass filter) and HPF (highpass filter)

import wave
import numpy as np
from scipy.signal import butter, lfilter, freqz
import tkinter as Tk    
import matplotlib.figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Specify wave file (uncomment one of the following)
wavfile = 'author.wav'
# wavfile = 'sines.wav'
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

# Block length : number of frames per block
# BLOCKLEN = 1024
BLOCKLEN = 512

BLOCK_DURATION = 1000.0 * BLOCKLEN/RATE # duration in milliseconds
print('Block length: %d' % BLOCKLEN)
print('Duration of block in milliseconds: %.2f' % BLOCK_DURATION)

def fun_quit():
  global CONTINUE
  print('Good bye')
  CONTINUE = False

# Filter Settings

ORDER = 3           # Filter order
fc_init = 0.1       # Initialize cut-off frequency
FILTER_TYPE = 'lowpass'
states = np.zeros(ORDER)

# DEFINE TKINTER COMPONENTS

root = Tk.Tk()    # Define root before creating figure

cutoff_freq = Tk.DoubleVar()            # Define Tk variable
cutoff_freq.set(fc_init)                   # Initilize

# Set up plotting...

fig1 = matplotlib.figure.Figure()
ax_input = fig1.add_subplot(3, 1, 1)
ax_freqresp = fig1.add_subplot(3, 1, 2)
ax_output = fig1.add_subplot(3, 1, 3)
fig1.set_size_inches((6, 8))  # (width, height)

n = np.arange(BLOCKLEN)
output_block = np.zeros(BLOCKLEN)

# input signal plot
[g_input] = ax_input.plot(n, output_block)
ax_input.set_ylim(-32000, 32000)
ax_input.set_xlim(0, BLOCKLEN)
ax_input.set_xlabel('Time (n)')
ax_input.set_title('Input signal')

# frequency response plot
[om, H] = freqz([0], [1])   # 'zero' filter to set om and H to initialize plot
f = om/(2 * np.pi)
[line_freqresp] = ax_freqresp.plot(f, np.abs(H))
ax_freqresp.set_xlim(0.5e-2, 0.5)
ax_freqresp.set_ylim(0, 1.2)
ax_freqresp.set_title('Frequency response')
ax_freqresp.set_xlabel('Frequency (cycles/sample)')
ax_freqresp.set_xscale('log')

# output signal plot
[g_output] = ax_output.plot(n, output_block)
ax_output.set_ylim(-32000, 32000)
ax_output.set_xlim(0, BLOCKLEN)
ax_output.set_xlabel('Time (n)')
ax_output.set_title('Output signal')

fig1.tight_layout()

my_canvas = FigureCanvasTkAgg(fig1, master = root)  # create canvas linked to GUI
C1 = my_canvas.get_tk_widget()          # create canvas widget
# my_canvas.draw()

# Update filter when slider is moved
def update_filter(event):
    global a, b
    fc = cutoff_freq.get()
    [b, a] = butter(ORDER, 2 * fc, btype = FILTER_TYPE)
    [om, H] = freqz(b, a)    
    line_freqresp.set_ydata(np.abs(H))

def toggle_filter_type():
    global FILTER_TYPE
    if FILTER_TYPE == 'lowpass':
        FILTER_TYPE = 'highpass'
    else:
        FILTER_TYPE = 'lowpass'
    update_filter(None)

S_cutoff = Tk.Scale(root, label = 'Cut-off frequency',
        variable = cutoff_freq, from_ = 0.01, to = 0.49, resolution = 0.01,
        command = update_filter)

B_toggle = Tk.Button(root, text = 'Toggle LPF/HPF', command = toggle_filter_type)
B_quit = Tk.Button(root, text = 'Quit', command = fun_quit)

# Place widgets
C1.pack()
S_cutoff.pack()
B_toggle.pack(fill = Tk.X)
B_quit.pack(fill = Tk.X)

update_filter(None)        # Run callback function to set filter coefficients a and b

CONTINUE = True

while CONTINUE:

    root.update()           # UPDATE TKINTER

    # Get block of samples from wave file
    input_bytes = wf.readframes(BLOCKLEN)

    if len(input_bytes) < WIDTH * BLOCKLEN:
        wf.rewind()                      # rewind to the start of the wave file
        input_bytes = wf.readframes(BLOCKLEN)

    # Convert binary data to number sequence (tuple)
    input_block = np.frombuffer(input_bytes, dtype = 'int16')

    # filtering
    [output_block, states] = lfilter(b, a, input_block, zi = states)

    g_input.set_ydata(input_block)
    g_output.set_ydata(output_block)
    my_canvas.draw()                  # display the figure
    # fig1.canvas.draw()                  # display the figure
    # Does it matter which draw function we use?

wf.close()

print('* Finished')
