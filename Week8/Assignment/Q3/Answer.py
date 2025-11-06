import pyaudio
import numpy as np
import tkinter as Tk    
from scipy.signal import lfilter

import matplotlib.figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import animation

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')

print('The matplotlib backend is %s' % matplotlib.get_backend())

# BLOCKLEN: Number of frames per block 
# BLOCKLEN   = 1024
# BLOCKLEN   = 512
BLOCKLEN   = 256
# BLOCKLEN   = 128
# BLOCKLEN   = 64
# BLOCKLEN   = 32

WIDTH       = 2         # Bytes per sample
CHANNELS    = 1         # Mono
RATE        = 8000      # Frames per second
MAXVALUE    = 2**15-1   # Maximum allowed output signal value (because WIDTH = 2)

BLOCK_DURATION = 1000.0 * BLOCKLEN/RATE # duration in milliseconds
print('Block length: %d' % BLOCKLEN)
print('Duration of block in milliseconds: %.2f' % BLOCK_DURATION)

# Number of notes in the octave
NUM_NOTES = 12
# Base frequency (Middle A)
f0 = 440.0
# Decay time (seconds), kept same for all notes
Ta = 1.2
ORDER = 2   # filter order
# --- End of Polyphony Setup ---

# Calculate frequencies for the full octave (12 notes)
frequencies = [f0 * (2**(k/12.0)) for k in range(NUM_NOTES)]
print(f'Frequencies (Hz) for the {NUM_NOTES} notes: {frequencies}')

##### Filter parameters for all notes

# Pole radius, kept same for all notes
r = 0.01 ** ( 1.0/(Ta * RATE) )       # 0.01 for 1 percent amplitude

# Calculate the filter coefficients (a_k, b_k) for each note
a_list = []
b_list = []
# States for each filter (12 independent state vectors)
states_list = [np.zeros(ORDER) for _ in range(NUM_NOTES)]
# Input signal for each filter (impulse at the middle of the block)
x_list = [np.zeros(BLOCKLEN) for _ in range(NUM_NOTES)]

for f in frequencies:
    om = 2.0 * np.pi * f / RATE
    # Filter coefficients (second-order recursive filter)
    a_k = [1, -2 * r * np.cos(om), r ** 2]
    b_k = [np.sin(om)]
    a_list.append(a_k)
    b_list.append(b_k)

# Map keyboard keys to notes (k=0 to k=11)
# You can customize this mapping
KEY_MAP = {
    'a': 0, 'w': 1, 's': 2, 'e': 3,
    'd': 4, 'f': 5, 't': 6, 'g': 7,
    'y': 8, 'h': 9, "u": 10, 'j': 11
}

# --- End of Filter Setup ---

# Open the audio output stream
p = pyaudio.PyAudio()
PA_FORMAT = pyaudio.paInt16
stream = p.open(
        format      = PA_FORMAT,
        channels    = CHANNELS,
        rate        = RATE,
        input       = False,
        output      = True,
        frames_per_buffer = BLOCKLEN)
# What if you specify a much larger value for frames_per_buffer? (try 8*BLOCKLEN)(latency)
# specify low frames_per_buffer to reduce latency

# --- Global state for keypresses ---
# Use a dictionary to track which note (by index k) was just pressed
KEYPRESS_NOTES = {} # Stores {k: True} for notes that need an impulse
# --- End of Global state ---

def my_function(event):
    global KEYPRESS_NOTES
    print('You pressed ' + event.char)
    if event.char == 'q':
        print('Good bye')
        root.quit()
    
    if event.char in KEY_MAP:
        # Mark the note index k to be played in the next block
        note_index = KEY_MAP[event.char]
        KEYPRESS_NOTES[note_index] = True
        print(f'Playing note k={note_index} at {frequencies[note_index]:.2f} Hz')


# Define Tkinter root

root = Tk.Tk()
root.bind("<Key>", my_function)

print('Press keys for sound. (Keys: ' + ', '.join(KEY_MAP.keys()) + ')')
print('Press "q" to quit')

# Define figure

my_fig = matplotlib.figure.Figure()
my_ax = my_fig.add_subplot(1, 1, 1)
# Plot will show the summed output signal
[g1] = my_ax.plot([], []) 
my_ax.set_ylim(-32000, 32000)
my_ax.set_xlim(0, BLOCKLEN * 1000.0 / RATE)   # Time axis in milliseconds 
my_ax.set_xlabel('Time (milliseconds)')
my_ax.set_title('Summed Output Signal (Chord)')

my_canvas = FigureCanvasTkAgg(my_fig, master = root)    # create Tk canvas from figure
C1 = my_canvas.get_tk_widget()    # canvas widget
C1.pack()                         # place canvas widget

# Define animation functions

M1 = np.int64(BLOCKLEN/2) # Location for the impulse

def my_init():
    t = np.arange(BLOCKLEN) * 1000/RATE
    g1.set_xdata(t)
    return (g1,)

def my_update(i):
    global states_list
    global x_list
    global KEYPRESS_NOTES
    
    # Initialize the total output signal for the current block
    y_total = np.zeros(BLOCKLEN)

    # 1. Apply impulse to inputs of the filters corresponding to pressed keys
    for k in range(NUM_NOTES):
        if k in KEYPRESS_NOTES and KEYPRESS_NOTES[k]:
            # Apply impulse to the input of the k-th filter
            x_list[k][M1] = 10000.0 # Impulse magnitude
            # Reset the keypress flag for this note
            KEYPRESS_NOTES[k] = False
        
        # 2. Filter the input signal x_k using the k-th filter's coefficients (a_k, b_k) and states
        [y_k, states_list[k]] = lfilter(b_list[k], a_list[k], x_list[k], zi = states_list[k])
        
        # 3. Sum the output of the k-th filter to the total output
        y_total = y_total + y_k
        
        # 4. Reset the impulse in the input for the next block
        x_list[k][M1] = 0.0        

    # 5. Process the total output signal
    y_total = np.clip(y_total, -MAXVALUE, MAXVALUE)     # Clipping
    g1.set_ydata(y_total)                         # update plot with the summed signal
    stream.write(y_total.astype('int16').tobytes(), BLOCKLEN)
    return (g1,)

my_anima = animation.FuncAnimation(
    my_fig,
    my_update,
    init_func = my_init,
    interval = 20,   # milliseconds (what happens if this is 200?)
    blit = True,
    cache_frame_data = False,
    repeat = False
)

Tk.mainloop()

# Close audio stream
stream.stop_stream()
stream.close()
p.terminate()

print('* Finished')