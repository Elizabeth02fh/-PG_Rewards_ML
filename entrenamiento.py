#from sklearn.datasets import load_digits  # dataset de imagenes de digitos
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA

from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
#import seaborn as sns
from skimage import io
import cv2
import os
import csv
import joblib

# clasificadores
# from sklearn.linear_model import LogisticRegression

data = []
labels = []
label = 0
wd = os.getcwd() + '\DATASET'
contenido = os.listdir(wd)
print(wd)
print(contenido)


def almacenar():
    global label
    for carpeta in contenido:
        # c = ''
        dir_carpeta = wd + '\\' + carpeta
        # print(dir_carpeta)
        print('Leyendo ' + carpeta)
        for imagen in os.listdir(dir_carpeta):
            # if os.path.isfile(dir_carpeta) and imagen.endswith('.jpg'):
            image = io.imread(dir_carpeta + "\\" + imagen)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # gray = cv2.resize(gray,(128,128))
            gray = gray.flatten()
            data.append(gray)
            labels.append(label)
            # writer.writerows(data)
            # print(imagen)
        label += 1
    print("finalizo")


def crearcsv():
    with open('dataset.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


almacenar()
crearcsv()

# ENTRENANDO

df = pd.read_csv('dataset.csv', header=None)
# X=np.array(df.drop([4096],axis=1))
X = df
# y=df[4096]
y = labels

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.90, random_state=0)

# n_components = 150
#pca = PCA(.95)
#pca.fit(x_train)
# pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(x_train)
#X_train_pca = pca.transform(x_train)
#X_test_pca = pca.transform(x_test)

regresionL = LogisticRegression(solver='lbfgs', max_iter=1000)
regresionL.fit(x_train, y_train)
joblib.dump(regresionL, 'modelo_entrenado_regresion.pkl')

maquinas = SVC()
# maquinas.fit(x_train,y_train)
maquinas.fit(x_train, y_train)
joblib.dump(maquinas, 'modelo_entrenado_SVM.pkl')

# kmedias = KMeans(n_clusters=2, random_state=0)
# kmedias.fit(X)

kvecinos = KNeighborsClassifier(n_neighbors=10, metric='minkowski')
kvecinos.fit(x_train, y_train)
joblib.dump(kvecinos, 'modelo_entrenado_KNN.pkl')

MLP = MLPClassifier(solver='lbfgs', max_iter=500, alpha=1e-5, hidden_layer_sizes=(128, 128, 128, 128, 128),
                    random_state=1)
MLP.fit(x_train, y_train)
joblib.dump(MLP, 'modelo_entrenado_MLP.pkl')

prediccionesL = regresionL.predict(x_test)
prediccionesM = maquinas.predict(x_test)
prediccionesKNN = kvecinos.predict(x_test)
prediccionesMLP = MLP.predict(x_test)

print(y_test)
print(prediccionesL)
print(prediccionesM)
print(prediccionesKNN)
print(prediccionesMLP)

score = regresionL.score(x_test, y_test)
score2 = maquinas.score(x_test, y_test)
score3 = kvecinos.score(x_test, y_test)
score4 = MLP.score(x_test, y_test)

mcl = metrics.confusion_matrix(y_test, prediccionesL)
print(mcl)
mcm = metrics.confusion_matrix(y_test, prediccionesM)
print(mcm)
mcknn = metrics.confusion_matrix(y_test, prediccionesKNN)
print(mcknn)
mcmpl = metrics.confusion_matrix(y_test, prediccionesMLP)
print(mcmpl)
# precision = precision_score(y_test, predicciones)
# exactitud = accuracy_score(y_test, predicciones)
# print("la presicion es: ", precision)
# print("la exactitud es: ", exactitud)
print("El score regresion es: ", score)
print("El score SVC es: ", score2)
print("El score KNN: ", score3)
print("El score MLP es: ", score4)
# output_array = np.array(data)
# np.savetxt("my_output_file.csv", output_array, delimiter=",")