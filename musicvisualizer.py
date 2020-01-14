import numpy as np
import librosa

import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

y, sr = librosa.load('Fanfare60.wav', sr=44100)

length = y.shape[0] / sr

n_fft = 4096

D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft+1))
D = D.T

frequencies = librosa.core.fft_frequencies(sr, n_fft=n_fft)

n_graphs = D.shape[0]
graph_time = length / n_graphs
n_graphs_sec = 1 // graph_time

fig = plt.figure()
ax = plt.axes()
line, = ax.plot([], [], lw=2)


def init():
    ax.set_xlim(0, 20000)
    ax.set_ylim(0, 700)
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data(frequencies, D[i])
    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_graphs, interval=length)
anim.save('CMajor.mp4')
plt.show()
