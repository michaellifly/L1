# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 22:20:00 2017

@author: Michael Li
"""


from sklearn import tree

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

#[height, weight, shoe size ]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]


Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# decision tree

clf1 = tree.DecisionTreeClassifier()

clf1 = clf1.fit(X,Y)

prediction1 = clf1.predict([[190,70,43]])



#SVC

clf2 = SVC(C=0.05)

clf2.fit(X,Y)

prediction2 =clf2.predict([[190,70,43]])


# GaussianNB

clf3 = GaussianNB()

clf3.fit(X,Y)

prediction3 =clf3.predict([[190,70,43]])
