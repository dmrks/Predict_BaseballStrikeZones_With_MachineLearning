import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

#1
#print(aaron_judge.columns)

#2
#print(aaron_judge.description.unique())

#3
#print(aaron_judge.type.unique())

#4
aaron_judge['type'] = aaron_judge['type'].map({'S':1, 'B':0})

#5
#print(aaron_judge.type.unique())

#6
#print(aaron_judge['plate_x'])
#print(aaron_judge['plate_z'])

#7
aaron_judge = aaron_judge.dropna(subset = ['plate_x', 'plate_z',"type"])

#8
plt.scatter(x =aaron_judge.plate_x, y= aaron_judge.plate_z, c=aaron_judge.type,cmap = plt.cm.coolwarm,alpha = 0.25)
training_set,validation_set =train_test_split(aaron_judge,random_state= 1)
classifier = SVC(kernel = 'rbf',gamma = 3,C = 1)
X = training_set[['plate_x', 'plate_z']]
y = training_set.type
classifier.fit(X,y)
draw_boundary(ax,classifier)
plt.show()
plt.clf()

#13 Acurracy = 0.8439073514602216
print(classifier.score(X,y))


