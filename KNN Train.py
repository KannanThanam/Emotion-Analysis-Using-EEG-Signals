import os
from skimage.transform import resize
from skimage.io import imread
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import roc_curve, auc

datadir='data' 
Categories=os.listdir(datadir)
flat_data_arr=[] #input array
target_arr=[] #output array

#path which contains all the categories of images
for i in Categories:
    
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(50,50,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)


df=pd.DataFrame(flat_data) #dataframe
df['Target']=target

x=df.iloc[:,:-1] #input data 
y=df.iloc[:,-1] #output data



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn import svm
from sklearn.model_selection import GridSearchCV



model = KNeighborsClassifier(n_neighbors=1)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=77,stratify=y)
print('Splitted Successfully')
model.fit(x_train,y_train)

filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))

y_pred=model.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")




probabilities = model.predict(x_test)
k=[]
print(probabilities)
for i in probabilities:
    if i>1:
        i=1
    k.append(i)
# select the probabilities for label 1.0
y_proba = list(k)

# calculate false positive rate and true positive rate at different thresholds
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)

# calculate AUC
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
# plot the false positive rate on the x axis and the true positive rate on the y axis
roc_plot = plt.plot(false_positive_rate,
                    true_positive_rate,
                    label='AUC = {:0.2f}'.format(roc_auc))

plt.legend(loc=0)
plt.plot([0,1], [0,1], ls='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate');
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


from sklearn.metrics import classification_report

print('\nClassification Report\n')
print(classification_report(y_test,y_pred))




confusion_mtx = confusion_matrix(y_test,y_pred) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()





