import torch
import torch.nn as nn
import numpy as np
from Util import Util

threshold = 0.62

class DataSet(torch.utils.data.Dataset):
    def __init__(self, x, labels):
        self.x = x
        self.labels = labels
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.labels[index]


class Model(nn.Module):
    def __init__(self, inputSize, hiddenSize, dropout = 0.2):
        super(Model, self).__init__()
        self.l1 = nn.Linear(inputSize, hiddenSize)
        self.hidden = nn.Linear(hiddenSize, hiddenSize+10)
        self.hidden2 = nn.Linear(hiddenSize+10, hiddenSize+10)
        self.hidden3 = nn.Linear(hiddenSize+10, hiddenSize)
        self.l2 = nn.Linear(hiddenSize, 1)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.l1(x)
        x = self.hidden(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.l2(x)
        x = self.activation(x)

        return x

def train(features, length, labels):
    model = Model(length, 10)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.02)
    numBatches = 100
    batchSize = len(features)//numBatches
   
    correct = 0
    epoch = 100
    falseNeg = 0
    falsePos = 0
    trueNeg = 0
    truePos = 0

    #start training
    model.train()
    print("Training...")
    for i in range(epoch):
        loader = iter(torch.utils.data.DataLoader(DataSet(features, labels), batch_size = batchSize, shuffle = True))
        loss = None
        for j in range(numBatches):
            x, l = next(loader)
            correct = 0
            optimizer.zero_grad()
            
            output = model(x)
            loss = criterion(torch.squeeze(output), torch.squeeze(l.float()))

            correct += (output == l.float()).sum()
            loss.backward()
            optimizer.step()

        print('Epoch {}'.format(i))

    return model

def CalcF1(tp, tn, fp, fn):
    if(tp == 0):
        return 0
    precision = tp/(tp + fp)
    recall = tp/(tp+fn)

    return 2*(precision*recall)/(precision + recall)

def Evaluate(model, processed1, labels):
    model.eval()
    criterion = torch.nn.BCELoss()
    correct = 0
    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0
    loss = None
    for i in range(len(processed1)):
        output = model(processed1[i])
        if output > threshold:
            if labels[i] == 1:
                truePos += 1
                correct += 1
            else:
                falsePos += 1
        else:
            if labels[i] == 0:
                trueNeg += 1
                correct += 1
            else:
                falseNeg += 1
        if(loss == None):
            loss = criterion(output.squeeze(), torch.tensor(labels[i]).float())
        else:
            loss.add_(criterion(output.squeeze(), torch.tensor(labels[i]).float()))
    
    f1 = CalcF1(truePos, trueNeg, falsePos, falseNeg)
    print('loss: {}  correct: {}  F1: {}'.format(loss.item()/len(processed1), correct/len(processed1), f1))

def PredictUnknown(model, features):
    model.eval()
    positive = 0
    negative = 0
    file = open("Data/KianArmandMcCollumAzadi_test_result.txt", "w")
    for i in range(len(features)):
        line = "test_id_"+str(i)+"\t"
        output = model(features[i])
        if output > threshold:
            line += "1\n"
            positive += 1
        else:
            line += "0\n"
            negative += 1
        
        file.write(line)
    file.close()

    print(positive)
    print(negative)


def main():
    vocab1, labels1, formatted1, formatted2 = Util.PreProcess("Data/train_with_label.txt", True)
    vocab2, labels2, formatted3, formatted4 = Util.PreProcess("Data/dev_with_label.txt", False)
    vocab3, labels3, formatted5, formatted6 = Util.PreProcess("Data/test_without_label.txt", False)
    trainedModel = None

    features1 = Util.AddFeatures(formatted1, formatted2, vocab1)
    features2 = Util.AddFeatures(formatted3, formatted4, vocab2)
    features3 = Util.AddFeatures(formatted5, formatted6, vocab3)

    trainedModel = train(features1, len(features1[0]), labels1)

    #check data on dev set
    Evaluate(trainedModel, features2, labels2)

    #evaluate and write predictions on test set
    PredictUnknown(trainedModel, features3)


main()