import re
import torch
from polyleven import levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class Vocab():
    def __init__(self):
        #these values set for the purpose of ensuring they're not used on accident by other words.
        self.sToI = {"<PAD>": 0, "": 1}
        self.iToS = {}
        self.frequencies = {}
        self.length = -1

    def RefineFreq(self, minFreq):
        for key, value in self.frequencies.items():
            if self.frequencies[key] < minFreq:
                if key in self.sToI.keys():
                    temp = self.sToI[key]
                    del self.sToI[key]
            else:
                if key not in self.sToI.keys():
                    self.sToI[key] = len(self.sToI)
        self.PopulateIToS()

    def PopulateIToS(self):
        for key,value in self.sToI.items():
            self.iToS[value] = key

    def NumToString(self, data):
        output = []
        for value in data:
            output.append(self.iToS[int(value)])

        return output
    
    def CalcFreq(self, data):
        #standardization of words of removing double spaces isolated by the .split()
        for sentence in data:
            for word in sentence.split(" "):
                word = word.strip()
                if word != "":
                    if word in self.frequencies:
                        self.frequencies[word] += 1
                    else:
                        self.frequencies[word] = 1
        
        if "" in self.frequencies:
            del self.frequencies[""]

    def MakeNumbers(self, data1, data2, labels, hasLabels, isTraining):
        output1 = []
        output2 = []
        newLabels = []

        length = 0
        
        #convert each word into an ID value for the tensor
        for i in range(len(data1)):
            temp1 = []
            temp2 = []
            for word in data1[i].split(" "):
                word = word.strip()
                #scrub out single letter words
                if word != "" and len(word) > 1:
                    temp1.append(self.sToI[word])

            if len(temp1) > self.length:
                self.length = len(temp1)

            for word in data2[i].split(" "):
                word = word.strip()
                if word != "":
                    temp2.append(self.sToI[word])

            if len(temp2) > self.length:
                self.length = len(temp2)
              
            #add the data in so even index is temp1, odd index is temp2
            output1.append(torch.FloatTensor(temp1))
            output2.append(torch.FloatTensor(temp2))
            if(hasLabels):
                newLabels.append(labels[i])
            if hasLabels and labels[i] == 1 and isTraining:
                output1.append(torch.FloatTensor(temp1))
                output2.append(torch.FloatTensor(temp2))
                output1.append(torch.FloatTensor(temp1))
                output2.append(torch.FloatTensor(temp2))
                newLabels.append(labels[i])
                newLabels.append(labels[i])
        return output1, output2, newLabels



class Util():
    @staticmethod
    def PreProcess(fileName, isTraining):
        f = open(fileName, "r", encoding="utf8")

        label = []
        test1 = []
        test2 = []
        vocab = {}

        line = f.readline()
        while line != "":
            temp = line.split("\t")
            if(fileName != "Data/test_without_label.txt"):
                label.append(int(temp[3]))
            test1.append(temp[1])
            test2.append(temp[2])

            line = f.readline()

        f.close()
        
        for i in range(len(label)):
            #remove case sensitivity
            test1[i] = test1[i].lower()
            test2[i] = test2[i].lower()

            #remove excess whitespace and punctuation via regex
            #some words are hyphenated, may need to change later
            test1[i] = re.sub(r'[^\w\s]', '', test1[i])
            test2[i] = re.sub(r'[^\w\s]', '', test2[i])

        #build up vocab
        vocab = Vocab()
        vocab.CalcFreq(test1)
        vocab.CalcFreq(test2)

        vocab.RefineFreq(1)
        processed1, processed2, label = vocab.MakeNumbers(test1, test2, label, len(label) > 0, isTraining)
        
        return vocab,label, processed1, processed2
    
    @staticmethod 
    def PadSize(length, data1, data2):
        for i in range(len(data1)):
            temp = data1[i].tolist()
            while len(temp) < length:   
                temp.append(0)
            data1[i] = torch.FloatTensor(temp)

            temp = data2[i].tolist()
            while len(temp) < length:    
                temp.append(0)
            data2[i] = torch.FloatTensor(temp)

    @staticmethod
    def AddFeatures(data1, data2, vocab):
        output = []
        for i in range(len(data1)):
            temp = []
            #list of strings
            tempSentence1 = vocab.NumToString(list(data1[i]))
            tempSentence2 = vocab.NumToString(list(data2[i]))
            
            #list of ints
            tempNums1 = list(data1[i])
            for i in range(len(tempNums1)):
                tempNums1[i] = int(tempNums1[i])

            tempNums2 = list(data2[i])
            for i in range(len(tempNums2)):
                tempNums2[i] = int(tempNums2[i])


            #FEATURE: number of words similar between them via word bag.
            #because there are so many words, there can be some astronomically high values. Will normalize between 0 and 1 to be more in range with other features
            values1 = sum(tempNums1)
            values2 = sum(tempNums2)
            total = values1 + values2
            values1 /= total
            values2 /= total

            temp.append(values1*2)
            temp.append(values2*2)

            #FEATURE: Levenshtein distance
            lev = levenshtein(" ".join(tempSentence1)," ".join(tempSentence2))

            #FEATURE: BLEU scores
            bleu = sentence_bleu(tempSentence1, tempSentence2, smoothing_function = SmoothingFunction().method4)

            temp.append(lev)

            output.append(torch.FloatTensor(temp))

        return output
