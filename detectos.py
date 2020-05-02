from scipy.io import wavfile as wav
from python_speech_features import mfcc, logfbank
#from scipy.fftpack import fft
import numpy as np

class Bin:
    def __init__(self, dataArray):
        self.dataArray = dataArray
        self.band = None

# Devuelve una lista con bins (los puros datos sin estado),
# que a la vez son listas con la cantidad de datos que conforma
# un bin: (rate/1000)*25
def binSegmentation(data, length_of_bin):
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
      output = []
      for b in bins:
            suma = 0
            for freq in b.dataArray:
                  suma+=freq
            avg = abs(suma/len(b.dataArray))
            if(avg < 10):
                  b.band = 0
            elif(avg >= 10 and avg < 100):
                  b.band = 1
            else:
                  b.band = 2

            output.append(b.band)
      return output

def getObservables(filename):
      rate, data = wav.read(filename)
      length_of_bin = int((rate/1000)*25)
      bins = binSegmentation(data, length_of_bin)
      return bandClassification(bins)


