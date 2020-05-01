from scipy.io import wavfile as wav
#from scipy.fftpack import fft
import numpy as np

rate, data = wav.read('cough1.wav')
length_of_bin = int((rate/1000)*25)
#fft_out = fft(data).real

class Bin:
    def __init__(self, dataArray, state):
        self.dataArray = dataArray
        self.state = state

# Devuelve una lista con bins (los puros datos sin estado),
# que a la vez son listas con la cantidad de datos que conforma
# un bin: (rate/1000)*25
def binSegmentation():
      totalBins = []
      true_index = 0
      for i in range(len(data)):
            i = true_index
            newBin = []
            for j in range(length_of_bin):
                  if(i >= len(data)):
                        return totalBins
                  
                  newBin.append(data[i])
                  i+=1

            true_index = i
            totalBins.append(newBin)
      
      return totalBins

bins_without_states = binSegmentation()