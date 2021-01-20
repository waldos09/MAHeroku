import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import mysql.connector as sql

db = sql.connect(
    host="bdpmye8dyhwt4hdqhgcp-mysql.services.clever-cloud.com",
    user="upk3pjsosofutpwj",
    passwd="CBl9dLCyWcsvG3wFSGov",
    database="bdpmye8dyhwt4hdqhgcp"
)
print(db)
cursor = db.cursor()
#cursor.execute('SELECT contestado FROM DEncuesta')
#c = cursor.fetchall()
cursor.execute('SELECT resultado FROM DEncuesta')
r = cursor.fetchall()
cursor.execute('SELECT valoracion FROM DEncuesta')
v = cursor.fetchall()
print(r)
print(v)
db.close()

#PARTE ESTEICA SIDEBAR Y TITULO                                                                                      
st.write("""
# Resultados Esperados
En este apartado se muestran tus posibles resultados proximos. Aquellos que veas mas bajos son posibles **Puntos Criticos** 
""")

st.sidebar.header('RESULTADOS DEL USUARIO')


#PARTE ESTEICA SIDEBAR Y TITULO

#IMPORTAR LOS DATOS DEL CSV

filename = 'Muestra.csv'
dataset = pd.read_csv(filename,header=0)

#PREPARAR LOS DATOS PARA ENTRENAR Y TESTEAR

#X_adr = dataset['RESULTADOS']
#X_adr = np.array(X_adr)
#X_adr = X_adr.reshape(-1,1)

#y_adr=dataset['VALORACION']

X_adr = r
X_adr = np.array(X_adr)
X_adr = X_adr.reshape(-1,1)

y_adr=v
print(X_adr)
print(y_adr)
#st.write('Tamaño de los datos:', X_adr.shape)


X_month = np.array(X_adr.shape)
X_monthF = X_month[0]/30
st.write('Numero de meses predecidos:',X_monthF)
#st.write('number of classes:', len(np.unique(y_adr)))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_adr,y_adr,test_size=0.2)

from sklearn.ensemble import RandomForestRegressor

#Defina el algoritmo a utilizar
adr = RandomForestRegressor(n_estimators =300,max_depth = 8)

#entrenar el modelo
adr.fit(X_train,y_train)

#realizo una prediccion
Y_pred = adr.predict(X_test)

#streamlit run app.py
scorepor = adr.score(X_train,y_train)*100
st.write('Machine Score:', scorepor)



fig = plt.figure()
X_grid = np.arange(min(X_test),max(X_test),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X_test,y_test)
plt.plot(X_grid,adr.predict(X_grid),color='red',linewidth=3)

plt.xlabel('Resultados Obtenidos')
plt.ylabel('Media de resultados')


st.pyplot(fig)
