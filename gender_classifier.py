from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

#Decision Tree

#[height, weight, shoe_size]
X = [[181,80,44], [177,70,43], [160,60,38], [154,54,37],
     [166,65,40], [190,90,47], [175,64,39], [177,70,40],
     [159,55,37], [171,75,42], [181,85,43]]

Y = ['male','female','female','female','male',
     'male','male','female','male','female','male']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

#k Nearest Neighbours
scaler = StandardScaler()
X = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X,Y)

#AdaBoost classifier
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
model = abc.fit(X,Y)

#make a prediction with each model
tree_prediction = clf.predict([[190,70,43]])
print(f"Decision tree prediction: {tree_prediction}")

neighbor_prediction = knn.predict([[190,70,43]])
print(f"kNN prediction: {neighbor_prediction}")

abc_prediction = model.predict([[190,70,43]])
print(f"AdaBoost prediction: {abc_prediction}")

