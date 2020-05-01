from scipy.io import wavfile as wav
#from scipy.fftpack import fft
import numpy as np

rate, data = wav.read('cough1.wav')
length_of_bin = int((rate/1000)*25)
#fft_out = fft(data).real

class Bin:
    def __init__(self, dataArray):
        self.dataArray = dataArray
        self.state = None
        self.low_band = None
        self.mid_band = None
        self.high_band = None

def printBinInfo(bins):
      print("Low Band\tMid Band\tHigh Band")
      for b in bins:
            print(b.low_band,"\t",b.mid_band,"\t",b.high_band)
# Devuelve una lista con bins (los puros datos sin estado),
# que a la vez son listas con la cantidad de datos que conforma
# un bin: (rate/1000)*25
def binSegmentation():
      totalBins = [] #array of bin objects
      true_index = 0
      for i in range(len(data)):
            i = true_index
            binData = []
            for j in range(length_of_bin):
                  if(i >= len(data)):
                        return totalBins
                  
                  binData.append(data[i])
                  i+=1
            binObj = Bin(binData)

            true_index = i
            totalBins.append(binObj)
      
      return totalBins

def bandClassification(bins):
      for b in bins:
            sum_low = 0
            sum_mid = 0
            sum_high = 0
            for freq in b.dataArray:
                  if freq < 2000:
                        sum_low+= freq
                  elif freq >= 2000 and freq < 4000:
                        sum_mid += freq
                  elif freq >= 4000 and freq < 22000:
                        sum_high += freq

            b.low_band = sum_low
            b.mid_band = sum_mid
            b.high_band = sum_high

bins_without_states = binSegmentation()
bandClassification(bins_without_states)
printBinInfo(bins_without_states)