import torch
import torch.nn as nn
import numpy as np
from Util import Util

class DataSet(torch.utils.data.Dataset):
    def __init__(self, x, y, labels):
        self.x = x
        self.y = y
        self.labels = labels
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.labels[index]

class Model(nn.Module):
    def __init__(self, inputSize, hiddenSize, dropout = 0.5):
        super(Model, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.connected1 = nn.Bilinear(self.inputSize,self.inputSize,self.hiddenSize)
        self.connected2 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.relu = nn.ReLU()
        self.connected3 = torch.nn.Linear(self.hiddenSize, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, y):
        hidden = self.connected1(x,y)
        relu = self.relu(hidden)
        hidden2 = self.dropout(relu)
        relu = self.relu(hidden2)
        hidden3 = self.connected2(relu)
        relu = self.relu(hidden3)
        output = self.connected3(relu)
        output = self.sigmoid(output)
        return output

def train(processed1, processed2, length, labels):
    model = Model(length, length)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    numBatches = 100
    batchSize = len(processed1)//numBatches

    sampler = torch.utils.data.WeightedRandomSampler([0.25,0.75], batchSize)
   
    correct = 0
    epoch = 100
    falseNeg = 0
    falsePos = 0
    trueNeg = 0
    truePos = 0

    #starting model
    model.eval()
    print("Before:")
    
    for i in range(len(processed1)):
        output = model(processed1[i], processed2[i])
        if output == 0:
            if labels[i] == 0:
                trueNeg += 1
            else:
                falseNeg += 1

        if output == 1:
            if labels[i] == 1:
                truePos += 1
            else:
                falseNeg += 1
        correct += (output == torch.tensor(labels[i])).float().sum()
    
    f1 = CalcF1(truePos, trueNeg, falsePos, falseNeg)
    print('correct: {}  F1: {}'.format(correct/len(processed1), f1))

    #start training
    model.train()
    print("Training...")
    for i in range(epoch):
        loader = iter(torch.utils.data.DataLoader(DataSet(processed1, processed2, labels), batch_size = batchSize, shuffle = True))
        loss = None
        for j in range(numBatches):
            x,y,l = next(loader)
            correct = 0
            optimizer.zero_grad()
        
            
            output = model(x,y)
            loss = criterion(torch.squeeze(output), torch.squeeze(l.float()))

            correct += (output == l.float()).sum()

        print('Epoch {}'.format(i))
        loss.backward()
        optimizer.step()



    #final model
    correct = 0
    epoch = 5
    falseNeg = 0
    falsePos = 0
    loss = None
    trueNeg = 0
    truePos = 0

    #Ending model
    model.eval()
    print("After:")
    
    for i in range(len(processed1)):
        output = model(processed1[i], processed2[i])
        if output == 0:
            if labels[i] == 0:
                trueNeg += 1
            else:
                falseNeg += 1

        if output == 1:
            if labels[i] == 1:
                truePos += 1
            else:
                falseNeg += 1
        if(loss == None):
            loss = criterion(output.squeeze(), torch.tensor(labels[i]).float())
        else:
            loss.add_(criterion(output.squeeze(), torch.tensor(labels[i]).float()))
        correct += (output == torch.tensor(labels[i])).float().sum()
    
    f1 = CalcF1(truePos, trueNeg, falsePos, falseNeg)
    print('loss: {}  correct: {}  F1: {}'.format(loss.item()/len(processed1), correct/len(processed1), f1))

    return model

def CalcF1(tp, tn, fp, fn):
    precision = tp/(tp + fp)
    recall = tp/(tp+fn)

    return 2*(precision*recall)/(precision + recall)

def Evaluate(model, processed1, processed2, labels):
    model.eval()
    criterion = torch.nn.BCELoss()
    correct = 0
    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0
    loss = None
    for i in range(len(processed1)):
        output = model(processed1[i], processed2[i])
        if output == 0:
            if labels[i] == 0:
                trueNeg += 1
            else:
                falseNeg += 1

        if output == 1:
            if labels[i] == 1:
                truePos += 1
            else:
                falseNeg += 1
        if(loss == None):
            loss = criterion(output.squeeze(), torch.tensor(labels[i]).float())
        else:
            loss.add_(criterion(output.squeeze(), torch.tensor(labels[i]).float()))
        correct += (output == torch.tensor(labels[i])).float().sum()
    
    f1 = CalcF1(truePos, trueNeg, falsePos, falseNeg)
    print('loss: {}  correct: {}  F1: {}'.format(loss.item()/len(processed1), correct/len(processed1), f1))


def main():
    length1, labels1, formatted1, formatted2 = Util.PreProcess("Data/train_with_label.txt")
    length2, labels2, formatted3, formatted4 = Util.PreProcess("Data/dev_with_label.txt")
    length3, labels3, formatted5, formatted6 = Util.PreProcess("Data/test_without_label.txt")
    trainedModel = None
    print(length1)
    print(length2)
    print(length3)

    #find longest sentence to normalize the remaining vectors to. The model will always train using the training set, but the size of the input size is what is affected here.
    if length1 >= length2 and length1 >= length3:
        Util.PadSize(length1, formatted3, formatted4)
        Util.PadSize(length1, formatted5, formatted6)
        trainedModel = train(formatted1, formatted2, length1, labels1)
    elif length2 >= length1 and length2 >= length3:
        Util.PadSize(length2, formatted1, formatted2)
        Util.PadSize(length2, formatted5, formatted6)
        trainedModel = train(formatted1, formatted2, length2, labels1)
    elif length3 >= length1 and length3 >= length2:
        Util.PadSize(length3, formatted3, formatted4)
        Util.PadSize(length3, formatted1, formatted2)
        trainedModel = train(formatted1, formatted2, length3, labels1)

    #check data on dev set
    Evaluate(trainedModel, formatted3, formatted4, labels2)

    #evaluate and write predictions on test set


main()