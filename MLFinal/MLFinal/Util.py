import re
import torch

class Vocab():
    def __init__(self):
        #these values set for the purpose of ensuring they're not used on accident by other words.
        self.sToI = {"<PAD>": 0, "": 1}     
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

    def MakeNumbers(self, data1, data2, labels, hasLabels):
        output1 = []
        output2 = []
        
        length = 0
        
        #convert each word into an ID value for the tensor
        for i in range(len(data1)):
            temp1 = []
            temp2 = []
            for word in data1[i].split(" "):
                word = word.strip()
                #scrub out single letter words except for "I", and all possessive articles that can be interchanged
                if word != "":
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
            
            if hasLabels and labels[i] == 1:
                output1.append(torch.FloatTensor(temp1))
                output2.append(torch.FloatTensor(temp2))
                labels.insert(i,1)
                
        return output1, output2, labels



class Util():
    @staticmethod
    def PreProcess(fileName):
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
        processed1, processed2, label = vocab.MakeNumbers(test1, test2, label, len(label) > 0)
        
        return vocab.length,label, processed1, processed2
    
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

