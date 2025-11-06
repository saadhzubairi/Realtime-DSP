import pyaudio, wave
import numpy as np
import numpy.typing as npt
import scipy.signal
import os
import math
import sys

# --- Configuration ---
# Use a relative path for the audio file. Make sure 'author.wav' is in the same directory
# Or change this line to point to your audio file.
WAV_FILENAME = 'author.wav' 
OUTPUT_WAV_FILENAME = 'author_output_scaled.wav'

# Audio Processing Parameters
BLOCKLEN = 2048         # FFT Block Size (must be a power of 2 for optimal FFT performance)
OVERLAP_FACTOR = 0.75   # Typical overlap factor (0.5 to 0.75 is common)
ALPHA = 1.2             # Initial Scaling factor (0.5 < alpha < 2)

# --- Core Processing Functions ---

def process_block_fft_scaling(input_block: npt.NDArray, alpha: float, blocklen: int) -> npt.NDArray:
    """
    Performs frequency scaling on a block using FFT, interpolation, and IFFT.
    
    Args:
        input_block: The windowed time-domain input block.
        alpha: The frequency scaling factor.
        blocklen: The size of the block.
        
    Returns:
        The windowed time-domain output block of size blocklen.
    """
    
    # 1. Forward FFT
    # Use rfft for real input, which returns only the non-redundant half of the spectrum.
    X = np.fft.rfft(input_block)
    fft_size_half = X.size # N/2 + 1
    
    # Initialize the output spectrum Y with zeros
    Y = np.zeros_like(X, dtype=np.complex128)
    
    # 2. Frequency Scaling via Interpolation
    for i in range(fft_size_half):
        # The new frequency index i corresponds to the original frequency index src_idx
        # src_idx = i / alpha (Scaling: new_freq = old_freq * alpha, so to find the 
        # source bin for a new bin i, we look at old_bin = i/alpha)
        src_idx = i / alpha
        
        # Check if the source index is within the valid range [0, fft_size_half - 1]
        if src_idx < fft_size_half - 1:
            # Linear Interpolation (resampling the spectrum)
            i_0 = int(np.floor(src_idx))
            i_1 = i_0 + 1
            t = src_idx - i_0 # fractional part for interpolation
            
            # Interpolate between the two nearest source bins
            Y[i] = (1.0 - t) * X[i_0] + t * X[i_1]
        # Frequencies scaled beyond the Nyquist limit (src_idx >= fft_size_half - 1) 
        # are naturally set to zero due to the initialization of Y.

    # 3. Inverse FFT
    # Use irfft to get the real-valued time-domain signal
    resultant = np.fft.irfft(Y, n=blocklen) 
    
    return resultant

# --- Main Execution and OLA Loop ---

try:
    # Get the directory of the script and construct the path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wavfile_path = os.path.join(base_dir, WAV_FILENAME)

    print(f'Attempting to open wave file: {wavfile_path}')

    # Open wave file (should be mono channel)
    wf = wave.open(wavfile_path, 'rb')
except FileNotFoundError:
    print(f"Error: The audio file '{WAV_FILENAME}' was not found in the script directory.")
    sys.exit(1)
except wave.Error as e:
    print(f"Error opening wave file: {e}")
    sys.exit(1)

# Read the wave file properties
CHANNELS = wf.getnchannels()
RATE = wf.getframerate()
signal_length = wf.getnframes()
WIDTH = wf.getsampwidth()
MAXVALUE = 2**(8 * WIDTH - 1) - 1 # Maximum allowed output signal value (e.g., 32767 for 16-bit)

print(f'\n--- Audio File Properties ---')
print(f'Channels: {CHANNELS}')
print(f'Frame Rate: {RATE} Hz')
print(f'Signal Length: {signal_length} frames')
print(f'Bytes per sample: {WIDTH}')
print(f'Initial Scaling Factor (ALPHA): {ALPHA}')

# --- Setup Overlap-Add (OLA) ---
# The number of frames to advance between blocks
HOP = math.floor(BLOCKLEN * (1 - OVERLAP_FACTOR))
OVERLAP = BLOCKLEN - HOP

# Hanning window for smooth tapering.
# Setting sym=False ensures a window size of BLOCKLEN, not BLOCKLEN+1
WINDOW = scipy.signal.windows.hann(BLOCKLEN, sym=False) 

# Initialise buffer for the overlapping output part.
output_buffer = np.zeros(OVERLAP, dtype=np.float64)

print(f'--- OLA Parameters ---')
print(f'Block Length (BLOCKLEN): {BLOCKLEN}')
print(f'Hop Size (HOP): {HOP} (frames written to output)')
print(f'Overlap Size (OVERLAP): {OVERLAP} (frames buffered)')

# --- PyAudio and Output File Setup ---
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(
    format=p.get_format_from_width(WIDTH),
    channels=CHANNELS,
    rate=RATE,
    input=False,
    output=True,
    frames_per_buffer=HOP # Set buffer size to HOP for efficiency
)

# Open output wave file
output_wf = wave.open(os.path.join(base_dir, OUTPUT_WAV_FILENAME), 'w')
output_wf.setframerate(RATE)
output_wf.setsampwidth(WIDTH)
output_wf.setnchannels(CHANNELS)

# --- The Real-Time Processing Loop ---
wf.rewind() # Start at the beginning of the file

print('\n* Starting real-time audio processing loop...')

# Use a larger buffer to read more data initially to allow for hopping
input_read_buffer = wf.readframes(BLOCKLEN)

while True:
    # Loop the audio file continuously
    if len(input_read_buffer) < WIDTH * BLOCKLEN:
        # Reached end of file, rewind and get the remaining data (if any)
        wf.rewind()
        
        # This simple rewind/read ensures continuous looping but may cause a slight pause
        # For a truly gapless loop, you'd crossfade the end and start blocks.
        remaining_data = input_read_buffer
        input_read_buffer = wf.readframes(BLOCKLEN)
        # For now, we'll just skip to the next loop iteration if the buffer is too small
        if len(remaining_data) < WIDTH * HOP:
            continue
            
    # Read the data for the current block and advance the file pointer by HOP
    # This simulates reading HOP frames every loop iteration
    binary_data = input_read_buffer[:WIDTH * BLOCKLEN]
    
    # Move the input buffer forward by HOP
    input_read_buffer = input_read_buffer[WIDTH * HOP:] + wf.readframes(HOP)
    
    # 1. Convert binary data to float/int
    input_block = np.frombuffer(binary_data, dtype=f'int{WIDTH*8}')
    
    # Normalize input for better floating point arithmetic (optional, but good practice)
    input_block_float = input_block.astype(np.float64) / MAXVALUE
    
    # 2. Apply windowing (CRITICAL for OLA)
    input_block_windowed = input_block_float * WINDOW
    
    # 3. Process the block (FFT Scaling)
    # The output is still windowed and needs to be overlapped-added
    output_block_windowed = process_block_fft_scaling(
        input_block=input_block_windowed, 
        alpha=ALPHA, 
        blocklen=BLOCKLEN
    ) * WINDOW # Apply the output window again

    # 4. Overlap-Add (CRITICAL for smooth output)
    
    # A. Get the non-overlapping part of the current output
    output_non_overlap = output_block_windowed[:HOP]
    
    # B. Add the previous block's overlap (from the buffer) to the current non-overlap
    output_audio_chunk = output_buffer[:HOP] + output_non_overlap
    
    # C. Update the buffer for the NEXT iteration
    # The new buffer contains the remainder of the old buffer plus the new overlap
    # New buffer size = OVERLAP
    new_overlap_data = output_block_windowed[HOP:]
    
    # Shift the old buffer by HOP frames (discard the part that was just output)
    new_buffer = np.zeros(OVERLAP, dtype=np.float64)
    new_buffer[:(OVERLAP - HOP)] = output_buffer[HOP:] 
    
    # Add the current block's overlap part
    new_buffer += new_overlap_data 
    output_buffer = new_buffer

    # 5. Denormalization, Clipping, and Formatting
    output_audio_chunk = output_audio_chunk * MAXVALUE
    output_audio_chunk = np.clip(output_audio_chunk, -MAXVALUE, MAXVALUE)
    output_audio_chunk = np.around(output_audio_chunk).astype(f'int{WIDTH*8}')

    # 6. Write to stream and file
    binary_data_out = output_audio_chunk.tobytes()

    stream.write(binary_data_out)
    output_wf.writeframes(binary_data_out)
    
    # Simulate stopping when there is no more data to read (for testing without loop)
    # if not wf.readframes(BLOCKLEN): break # Uncomment this to stop after one pass

print('* Finished audio loop')

# --- Cleanup ---
# Handle any remaining data in the buffer before closing
if OVERLAP > 0:
    # Output the remaining overlap data (which is now in the buffer)
    final_output = output_buffer
    final_output = final_output * MAXVALUE
    final_output = np.clip(final_output, -MAXVALUE, MAXVALUE)
    final_output = np.around(final_output).astype(f'int{WIDTH*8}')
    
    binary_data_final = final_output.tobytes()
    stream.write(binary_data_final)
    output_wf.writeframes(binary_data_final)

stream.stop_stream()
stream.close()
p.terminate()

wf.close()
output_wf.close()
print(f'Output written to {OUTPUT_WAV_FILENAME}')