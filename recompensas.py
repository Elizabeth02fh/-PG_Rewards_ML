from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import uic
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap
import sys
import os
import joblib
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
faceClassif = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
path = ""
modelo = joblib.load('modelo_entrenado_SVM.pkl')

rostros = []
cantidad_rostros = 0
req_encontrado = 0
resolucion = 0

class ejemplo_GUI(QMainWindow):
     def __init__(self):
         super().__init__()
         uic.loadUi(self.resource_path('interfaz.ui'), self)
         self.btn_cargar_foto.clicked.connect(self.selectFile)
         self.btn_abrir_camara.clicked.connect(self.abrirCamara)
         self.btn_procesar.clicked.connect(self.procesarRostros)
         self.setWindowIcon(QIcon(self.resource_path('icon.png')))

     def resource_path(self, relative_path):
         if hasattr(sys, '_MEIPASS'):
             return os.path.join(sys._MEIPASS, relative_path)
         return os.path.join(os.path.abspath("."), relative_path)

     def selectFile(self):
         global path
         global resolucion
         path = str(QFileDialog.getOpenFileName()[0])
         try:
             self.lbl_resultado.setText("")
             self.lbl_resultado.setStyleSheet("background-color:yellow;color:white")
             img_res = cv.imread(path)
             hol = img_res.shape
             resolucion = hol[0]
             img_res = cv.resize(img_res, (351, 311))
             cv.imwrite('hi.jpg', img_res)
             print("guardado")
             pixmap = QPixmap("hi.jpg")
             self.lbl_imagen.setPixmap(pixmap)
             self.verificarRostros()
         except:
             pass

     def verificarRostros(self):
         global path
         global faceClassif
         #global modelo
         global cantidad_rostros
         print("escribir rostros")
         image = cv.imread(path)
         gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
         faces = faceClassif.detectMultiScale(gray, 1.1, 5)
         count = 0

         for (x, y, w, h) in faces:
             count = count + 1
         print("rostros encontrados ", count)
         cantidad_rostros = count

     def procesarRostros(self):
         global path
         global modelo
         global cantidad_rostros
         global req_encontrado
         global resolucion
         global faceClassif

         try:
             image = cv.imread(path)
             imageAux = image.copy()
             gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
             faces = faceClassif.detectMultiScale(gray, 1.1, 5)
             count = 0

             if resolucion == 128:
                print("resolucion 128 ")
                self.procesar()
             else:
                 if cantidad_rostros == 1:
                     print("cantidad rostros uno")

                     for (x, y, w, h) in faces:
                         cv.rectangle(image, (x, y), (x + w, y + h), (128, 0, 255), 2)
                         rostro = imageAux[y:y + h, x:x + w]
                         rostro = cv.resize(rostro, (128, 128), interpolation=cv.INTER_CUBIC)
                         gris = cv.cvtColor(rostro, cv.COLOR_BGR2GRAY)
                         X = np.array(gris).reshape(1, -1)
                         try:
                             persona = modelo.predict(X)
                             print(persona[0])
                             persona = int(persona[0])
                             if (persona == 3):
                                 self.lbl_resultado.setText("INOCENTE")
                                 self.lbl_resultado.setStyleSheet("background-color:green;color:white")
                             else:
                                 self.lbl_resultado.setText("REQUISITORIADO ")
                                 self.lbl_resultado.setStyleSheet("background-color:red;color:white")
                         except:
                             print("dio error")

                 elif cantidad_rostros > 1:
                     print("cantidad rostros mas de uno")

                     for (x, y, w, h) in faces:
                         cv.rectangle(image, (x, y), (x + w, y + h), (128, 0, 255), 2)
                         rostro2 = imageAux[y:y + h, x:x + w]
                         rostro = cv.resize(rostro2, (128, 128), interpolation=cv.INTER_CUBIC)
                         gris = cv.cvtColor(rostro, cv.COLOR_BGR2GRAY)
                         X = np.array(gris).reshape(1, -1)
                         try:
                             persona = modelo.predict(X)
                             print(persona[0])
                             persona = int(persona[0])
                             if (persona == 3):
                                 pass
                             else:
                                 req_encontrado += 1
                                 img_res = cv.resize(rostro2, (351, 311))
                                 cv.imwrite('hi.jpg', img_res)
                                 pixmap = QPixmap("hi.jpg")
                                 self.lbl_imagen.setPixmap(pixmap)
                         except:
                             print("dio error")
                     if req_encontrado >= 1:
                         self.lbl_resultado.setText("REQUISITORIADO ENCONTRADO")
                         self.lbl_resultado.setStyleSheet("background-color:red;color:white")

                     else:
                         self.lbl_resultado.setText("INOCENTES")
                         self.lbl_resultado.setStyleSheet("background-color:green;color:white")
                 else:
                     self.lbl_resultado.setText("NO HAY ROSTROS")
         except:
            pass
         print("escribir rostros")
         req_encontrado = 0

     def abrirCamara(self):
         global faceClassif
         global path
         print("abriendo camara...")
         cap = cv.VideoCapture(0, cv.CAP_DSHOW)
         count = 0
         cont = 1
         while True:
             ret, frame = cap.read()
             frame = cv.flip(frame, 1)
             gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
             auxFrame = frame.copy()
             faces = faceClassif.detectMultiScale(gray, 1.3, 5)
             k = cv.waitKey(1)
             if k == 27:
                 break
             if cont == 2:
                 break

             for (x, y, w, h) in faces:
                 cv.rectangle(frame, (x, y), (x + w, y + h), (128, 0, 255), 2)
                 rostro = auxFrame[y:y + h, x:x + w]
                 rostro = cv.resize(rostro, (128, 128), interpolation=cv.INTER_CUBIC)
                 if k == ord('p'):
                     cv.imwrite('hi.jpg', rostro)
                     img_res = cv.imread('hi.jpg')
                     img_res = cv.resize(img_res, (351, 311))
                     cv.imwrite('hi.jpg', img_res)
                     print("guardado")
                     pixmap = QPixmap("hi.jpg")
                     self.lbl_imagen.setPixmap(pixmap)
                     gris = cv.cvtColor(cv.resize(rostro, (128, 128)), cv.COLOR_BGR2GRAY)
                     X = np.array(gris).reshape(1, -1)
                     print("procesando...")
                     print(X.shape)
                     try:
                         persona = modelo.predict(X)
                         print(persona[0])
                         persona = int(persona[0])
                         if (persona == 3):
                             self.lbl_resultado.setText("INOCENTE")
                             self.lbl_resultado.setStyleSheet("background-color:green;color:white")
                         else:
                             self.lbl_resultado.setText("REQUISITORIADO")
                             self.lbl_resultado.setStyleSheet("background-color:red;color:white")
                     except:
                         print("dio error")
                     cont = 2
                     #count = count + 1
             cv.rectangle(frame, (10, 5), (450, 25), (255, 255, 255), -1)
             cv.putText(frame, 'Presione p, para procesar la imagen', (10, 20), 2, 0.5, (128, 0, 255),1, cv.LINE_AA)
             cv.imshow('frame', frame)
         print("salio de while")
         path = ""
         cap.release()
         cv.destroyAllWindows()

     def procesar(self):
         global path
         global modelo
         print(path)
         img = cv.imread(path)
         gris = cv.cvtColor(cv.resize(img, (128, 128)), cv.COLOR_BGR2GRAY)
         X = np.array(gris).reshape(1, -1)
         print(X.shape)
         try:
            persona = modelo.predict(X)
            print(persona[0])
            persona = int(persona[0])
            if(persona == 3):
                self.lbl_resultado.setText("INOCENTE")
                self.lbl_resultado.setStyleSheet("background-color:green;color:white")
            else:
                self.lbl_resultado.setText("REQUISITORIADO ")
                self.lbl_resultado.setStyleSheet("background-color:red;color:white")
         except:
            print("dio error")

if __name__ == '__main__':
     app = QApplication(sys.argv)
     GUI = ejemplo_GUI()
     GUI.show()
     sys.exit(app.exec_())