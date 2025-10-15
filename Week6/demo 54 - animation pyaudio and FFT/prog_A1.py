# plot_wave_file_and_play_using_animate.py

import pyaudio
import struct
import wave
import matplotlib
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')      # Optional if using Mac OSX

print('The matplotlib backend is %s' % pyplot.get_backend())  # The backend used plotting

# Specify wave file
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

# Block length (number of frames per block)
BLOCKLEN = 256
# BLOCKLEN = 128
# BLOCKLEN = 64
# ? ? How short can you make the block without having audible artifacts ?

Num_Plots = int(LEN/BLOCKLEN) - 1
print('Number of frames = ', Num_Plots)


# Open the audio output stream
p = pyaudio.PyAudio()

PA_FORMAT = p.get_format_from_width(WIDTH)
stream = p.open(
    format = PA_FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = False,
    output = True,
    frames_per_buffer = BLOCKLEN)   # optional, but can help to reduce latency

# Set up plotting...

fig1 = pyplot.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)
[g1] = ax1.plot([], [])

def my_init():
    print('hello')
    g1.set_xdata(range(BLOCKLEN))
    g1.set_ydata(BLOCKLEN * [0])
    ax1.set_ylim(-32000, 32000)
    ax1.set_xlim(0, BLOCKLEN)
    return (g1,)

def my_update(i):
    # Get block of samples from wave file
    input_bytes = wf.readframes(BLOCKLEN)
    
    # Convert binary data to number sequence (tuple)
    signal_block = struct.unpack('h' * BLOCKLEN, input_bytes)

    g1.set_ydata(signal_block)

    # Write binary data to audio output stream
    stream.write(input_bytes, BLOCKLEN)

    return (g1,)


my_anima = FuncAnimation(
    fig1,
    my_update,
    frames = Num_Plots,
    init_func = my_init,
    interval = 10,   # milliseconds (what happens if this is 200?)
    blit = True,
    repeat = False
)
pyplot.show()   # Needed for FuncAnimation to show plots

stream.stop_stream()
stream.close()
p.terminate()
wf.close()

print('* Finished')


