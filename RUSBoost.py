from sklearn import svm
from sklearn import tree
from math import log
import random

class RUSBoost:


    def __init__(self, instances, labels, base_classifier, n_classifier, balance):
        
        self.w_update=[]
        self.clf = []
        self.n_classifier = n_classifier
        for i in range(n_classifier):
            self.clf.append(base_classifier)
        self.rate = balance
        self.X = instances
        self.Y = labels
        
        # initialize weight
        self.weight = []
        self.init_w = 1.0/len(self.X)
        for i in range(len(self.X)):
            self.weight.append(self.init_w)
    
    def classify(self, instance):
        
        positive_score = 0 # in case of +1
        negative_score = 0 # in case of 0
        
        for k in range(self.n_classifier):
            if self.clf[k].predict(instance) == 1:
                positive_score += log(1/self.w_update[k])
            else:
                negative_score += log(1/self.w_update[k])
        if negative_score <= positive_score:
            return 1
        else:
            return 0
        
            
        
    def learning(self):
        
        k = 0
        while k < self.n_classifier:
            
            sampled = self.undersampling()
            sampled_X = []
            sampled_Y = []
            sampled_weight = []
            
            for s in sampled:
                sampled_X.append(s[1])
                sampled_Y.append(s[2])
                sampled_weight.append(self.weight[s[0]])
                
            self.clf[k].fit(sampled_X, sampled_Y, sampled_weight)
           
   
            loss = 0
            for i in range(len(self.X)):
                if self.Y[i] == self.clf[k].predict(self.X[i]):
                    continue
                else:
                    loss += self.weight[i]
    
            self.w_update.append(loss/(1-loss))
        
            for i in range(len(self.weight)):
                if loss == 0:
                    self.weight[i] = self.weight[i]
                elif self.Y[i] == self.clf[k].predict(self.X[i]):
                    self.weight[i] = self.weight[i] * (loss / (1 - loss))
                       
            sum_weight = 0
            for i in range(len(self.weight)):
                sum_weight += self.weight[i]
              
            for i in range(len(self.weight)):
                self.weight[i] = self.weight[i] / sum_weight
            k = k + 1
     
            
    def undersampling(self):
        
        '''Check the major class'''
        diff = self.Y.count(1) > self.Y.count(0)
        delete_list = []
        keep_list =[]
        if  diff:
            for i in range(len(self.Y)):
                if self.Y[i] == 1:
                    delete_data = [i, self.X[i], 1]
                    delete_list.append(delete_data)
                else:
                    keep_data = [i, self.X[i], 0]
                    keep_list.append(keep_data)
        else:
            for i in range(len(self.Y)):
                if self.Y[i] == 0:
                    delete_data = [i, self.X[i], 0]
                    delete_list.append(delete_data)
                else:
                    keep_data = [i, self.X[i], 1]
                    keep_list.append(keep_data)  
        
        while len(delete_list) > self.rate*(len(delete_list)+len(keep_list)):
            k = random.choice(range(len(delete_list)))
            delete_list.pop(k)
        
        all_list = delete_list + keep_list
        return sorted(all_list, key=lambda x:x[2])
    
if __name__ == '__main__':
    
    ''' 
    Choose a base classifier, e.g. SVM, Decision Tree
    '''
    #base_classifier = svm.SVC()
    base_classifier = tree.DecisionTreeClassifier()

    '''
    Set the number of base classifiers
    ''' 
    N = 20
    
    '''
    Set the rate of minor instances to the total instances
    If the rate is 0.5, the numbers of both instances become equal
    by random under sampling. 
    ''' 
    rate = 0.05
    
    
    '''
    Preparation of data
        "Ecoli data" from
        http://www.cs.gsu.edu/~zding/research/benchmark-data.php
        Test data: randomly selected one
        Supervised data: the others 
    '''
    supervisedData =[]
    supervisedLabel=[]
    testData =[]
    testLabel =[]
    n_features = 7
    
    # Read from a text file
    lines = [] ## all lines
    for line in open('x1data.txt', 'r'):
        elements = line[:-1].split(' ')
        lines.append(elements)
    
    # Select a line for test data
    selectline = lines.pop(random.randint(0, len(lines)))
    
    # Separate each line into a feature vector and a lebel
    testLabel.append(int(selectline[0]))
    testData = [0]*n_features
    for i in range(1,n_features+1):
        pair = selectline[i].split(':')
        if pair[0] != '':
            testData[int(pair[0])-1] = pair[1]
    
    for line in lines:
        supervisedLabel.append(int(line[0]))
        sData = [0]*n_features
        for i in range(1,n_features+1):
            pair = line[i].split(':')
            if pair[0] != '':
                sData[int(pair[0])-1] = pair[1]
        supervisedData.append(sData)
    
    
    rus = RUSBoost(supervisedData, supervisedLabel, base_classifier, N, rate)
    rus.learning()
    classifiedLabel = rus.classify(testData)
    print "Test data:",
    for t in testData:
        print t,
    print
    print "Classified to :" + str(classifiedLabel)
    print "True label is :" + str(testLabel[0])
    