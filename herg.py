# -*- coding: utf-8 -*-
"""
Created on Fri Sep 04 12:59:09 2015

@author: Chris Arthur

"""

import sys, cPickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn import preprocessing
import random
 
min_max_scaler = preprocessing.MinMaxScaler()

'''
Open pre-formatted Chembl data file First column = structure. Second column = pIC50
Split data into test and training sets
'''

f = open('herg2.csv', 'r')

data = []

test, train, testTarget, trainTarget =[],[],[],[]

for line in f:
    line.replace('\n','')
    temp=line.split(',')
    smiles=temp[0]
    pIC50 = float(temp[1])
    if random.randint(0,3) == 2:
		#Generate molecule from smiles string.
        test.append(Chem.MolFromSmiles(smiles))
        testTarget.append(pIC50)
    else:
        train.append(Chem.MolFromSmiles(smiles))
        trainTarget.append(pIC50)        
    
def ClusterFps(fps,cutoff=0.2):
	# Function to group molecules based on their similarity to other compounds in the HERG set.
	# This is an attempt to overcome some of the problems associated with not
	# having a single series of related compounds
	
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1,nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists,nfps,cutoff,isDistData=True)
    return cs
    



'''
Use RDKit to calculate molecular descriptors from the Smiles string
'''

nms=[x[0] for x in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
print(len(nms))
 
trainDescrs = [calc.CalcDescriptors(x) for x in train]
from rdkit.Chem import AllChem

'''
Cluster the compounds together
'''

fps = [AllChem.GetMorganFingerprintAsBitVect(x,2,1024) for x in train]
clusters=ClusterFps(fps,cutoff=0.4)
print "Number of clusters", len(clusters)

clusterCenters=[]
for x in range(len(clusters)):
    clusterCenters.append( clusters[x][0] )

	
def distij(i,j,fps=fps):
	# Function to calculate the distance similarity to cluster centers
    return 1-DataStructs.DiceSimilarity(fps[i],fps[j])


testDescrs  = [calc.CalcDescriptors(x) for x in test]

# Get into a form ready for using SciKit-Learn

trainDescrs = np.array(trainDescrs)
testDescrs = np.array(testDescrs)
 
x_train_minmax = min_max_scaler.fit_transform( trainDescrs )
x_test_minmax = min_max_scaler.fit_transform( testDescrs )
 
import sys, cPickle
from sklearn.ensemble import RandomForestRegressor

train_x, train_y = trainDescrs,  trainTarget
test_x, test_y = testDescrs, testTarget

test_y=np.asarray(test_y)

print "RANDOMFOREST"
nclf = RandomForestRegressor( n_estimators=200, max_depth=5, random_state=0, n_jobs=-1 )
nclf = nclf.fit( train_x, train_y )
preds = nclf.predict( test_x )

from sklearn.metrics import r2_score
r2 = r2_score(test_y, preds)
print r2
mse = np.mean((test_y - preds)**2)
print mse
#print metrics.confusion_matrix(test_y, preds)
#print metrics.classification_report(test_y, preds)
accuracy = nclf.score(test_x, test_y)
print accuracy
 
import pylab as pl
 
fig,ax = pl.subplots()
ax.scatter(test_y, preds, alpha=0.3)
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.plot( label="r^2=" + str(r2), c="r")
ax.legend(loc="lower right")

fig.show()

# 
#from sklearn.svm import SVR
#
################################################################################
## Fit regression model
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)
#y_rbf = svr_rbf.fit(train_x, train_y).predict(test_x)
#y_lin = svr_lin.fit(train_x, train_y).predict(test_x)
#y_poly = svr_poly.fit(train_x, train_y).predict(test_x)
#
#r2 = r2_score(test_y, y_rbf)
#print r2
#mse = np.mean((test_y - y_rbf)**2)
#print mse
##print metrics.confusion_matrix(test_y, preds)
##print metrics.classification_report(test_y, preds)
#accuracy = svr_rbf.score(test_x, test_y)
#print accuracy
#
#pl.scatter(test_y, y_rbf)
#pl.plot( label="r^2=" + str(r2), c="r")
#pl.legend(loc="lower right")
#pl.title("SVR RBF")
#pl.show()
#
#r2 = r2_score(test_y, y_lin)
#print r2
#mse = np.mean((test_y - y_lin)**2)
#print mse
##print metrics.confusion_matrix(test_y, preds)
##print metrics.classification_report(test_y, preds)
#accuracy = svr_lin.score(test_x, test_y)
#print accuracy
#
#pl.scatter(test_y, y_lin)
#pl.plot( label="r^2=" + str(r2), c="r")
#pl.legend(loc="lower right")
#pl.title("SVR linear")
#pl.show()
#
#r2 = r2_score(test_y, y_poly)
#print r2
#mse = np.mean((test_y - y_poly)**2)
#print mse
##print metrics.confusion_matrix(test_y, preds)
##print metrics.classification_report(test_y, preds)
#accuracy = svr_poly.score(test_x, test_y)
#print accuracy
#
#
#pl.scatter(test_y, y_poly)
#pl.plot( label="r^2=" + str(r2), c="r")
#pl.legend(loc="lower right")
#pl.title("SVR poly")
#pl.show()
#

from sklearn import linear_model
clf = linear_model.Ridge (alpha = 100)
ridge = clf.fit(train_x, train_y).predict(test_x)

r2 = r2_score(test_y, ridge)
print r2
mse = np.mean((test_y - ridge)**2)
print mse
#print metrics.confusion_matrix(test_y, preds)
#print metrics.classification_report(test_y, preds)


fig,ax = pl.subplots()
ax.scatter(test_y, ridge, alpha=0.3)
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.plot( label="r^2=" + str(r2), c="r")
ax.legend(loc="lower right")

fig.show()

clf = linear_model.Lasso(alpha = 0.1)
lasso = clf.fit(train_x, train_y).predict(test_x)

r2 = r2_score(test_y, lasso)
print r2
mse = np.mean((test_y - lasso)**2)
print mse
#print metrics.confusion_matrix(test_y, preds)
#print metrics.classification_report(test_y, preds)


fig,ax = pl.subplots()
ax.scatter(test_y, lasso, alpha=0.3)
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.plot( label="r^2=" + str(r2), c="r")
ax.legend(loc="lower right")

fig.show()

clf = linear_model.BayesianRidge()

bayesRidge = clf.fit(train_x, train_y).predict(test_x)

r2 = r2_score(test_y, bayesRidge)
print r2
mse = np.mean((test_y - bayesRidge)**2)
print mse
#print metrics.confusion_matrix(test_y, preds)
#print metrics.classification_report(test_y, preds)


fig,ax = pl.subplots()
ax.scatter(test_y, bayesRidge, alpha=0.3)
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.plot( label="r^2=" + str(r2), c="r")
ax.legend(loc="lower right")

fig.show()


from sklearn.ensemble import GradientBoostingRegressor

clf = GradientBoostingRegressor(n_estimators=50)

gbr = clf.fit(train_x, train_y).predict(test_x)

r2 = r2_score(test_y, gbr)
print r2
mse = np.mean((test_y - gbr)**2)
print mse
#print metrics.confusion_matrix(test_y, preds)
#print metrics.classification_report(test_y, preds)


pl.scatter(test_y, gbr)
pl.plot( label="r^2=" + str(r2), c="r")
pl.legend(loc="lower right")
pl.title("Gradient boosting regression")
pl.show()



from sklearn.ensemble import AdaBoostRegressor

clf = AdaBoostRegressor(n_estimators=100)

ada = clf.fit(train_x, train_y).predict(test_x)

r2 = r2_score(test_y, ada)
print r2
mse = np.mean((test_y - ada)**2)
print mse
#print metrics.confusion_matrix(test_y, preds)
#print metrics.classification_report(test_y, preds)


fig,ax = pl.subplots()
ax.scatter(test_y, ada, alpha=0.3)
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.plot( label="r^2=" + str(r2), c="r")
ax.legend(loc="lower right")

fig.show()

from sklearn.ensemble import ExtraTreesRegressor

clf = ExtraTreesRegressor(n_estimators=200)

clf.fit(train_x, train_y)


importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
                                 
# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

mean = np.asarray(importances).mean()
transformTrainX=clf.transform(train_x, threshold = mean)
transformTestX=clf.transform(test_x, threshold = mean)
clf.fit(transformTrainX, train_y)
et = clf.predict(transformTestX)

r2 = r2_score(test_y, et)
print r2
mse = np.mean((test_y - et)**2)
print mse


fig,ax = pl.subplots()
ax.scatter(test_y, et, alpha=0.3)
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.plot( label="r^2=" + str(r2), c="r")
ax.legend(loc="lower right")

fig.show()


# Regression using SciKit-NeuralNetwork

from sknn.mlp import Regressor, Layer

nn = Regressor(
    layers=[
        Layer("Rectifier", units=100),
        Layer("Linear")],
    learning_rate=0.0000001,
    n_iter=40
    )
    
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
test_x = np.asarray(test_x)
test_y = np.asarray(test_y)

nn.fit(train_x, train_y )
print "Results of SKNN Regression...."
print "==============================\n"

preds = nn.predict(test_x)
# The coefficients
#print('Coefficients: ', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((preds - test_y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % nn.score(test_x, test_y))

import matplotlib.pyplot as plt

# Plot outputs
#plt.scatter(test_x, test_y,  color='black')
plt.scatter(test_y, preds, color='blue')


plt.xticks(())
plt.yticks(())

plt.show()

