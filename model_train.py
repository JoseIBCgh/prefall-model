import pandas as pd
import numpy as np

import os
import io

import warnings

import libraries

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

import pickle

'''**********************************************************VARIABLES_INICIALES*****************************************************'''

#Variable para mostrar por pantalla algunos resultados. Por defecto no
verbose=False

#Ruta base de los ficheros para el entrenamiento
base="./datos/CTIC_casos/"

#Leemos el fichero csv con el resumen de los casos
df = pd.read_csv(base+"CasosPrefall.csv").dropna()
if verbose:
    print("Dataframe resumen cargado")
    print(df)

'''**********************************************************FUNCIONES*****************************************************'''



def train_model_normal(per_train,df_train_test):  
    
    #Porcentaje de datos para entrenamiento
    row_train=int(per_train * len(df_train_test))

    #Conjunto de datos para CV
    X=df_train_test.drop("riesgo",axis=1).to_numpy()
    y=df_train_test['riesgo']

    #Conjunto de datos de train
    df_train=df_train_test.iloc[0:row_train]
    X_train=df_train.drop("riesgo",axis=1).to_numpy()
    y_train=df_train['riesgo'].to_numpy()

    #Conjunto de datos de test
    df_test=df_train_test.iloc[row_train:]
    X_test=df_test.drop("riesgo",axis=1).to_numpy()
    y_test=df_test['riesgo'].to_numpy()

    #Seleccionamos varios modelos para elegir el mejor.
    modelos=[]
    modelo=RandomForestClassifier()
    modelos.append(modelo)
    modelo=XGBClassifier(base_score=None, booster='gblinear', callbacks=None,
                  colsample_bylevel=None, colsample_bynode=None,
                  colsample_bytree=None, early_stopping_rounds=None,
                  enable_categorical=False, eta=0.3, eval_metric=None,
                  feature_types=None, gamma=None, gpu_id=None, grow_policy=None,
                  importance_type=None, interaction_constraints=None,
                  learning_rate=None, max_bin=None, max_cat_threshold=None,
                  max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
                  max_leaves=None, min_child_weight=None,
                  monotone_constraints=None, n_estimators=50, n_jobs=None,
                  nthread=4, num_parallel_tree=None)
    modelos.append(modelo)
    modelo=LogisticRegression(C=1000.0,penalty='l2')
    modelos.append(modelo)
    modelo=SVC(C=0.1,gamma=1,kernel='rbf')
    modelos.append(modelo)
    modelo=DecisionTreeClassifier()
    modelos.append(modelo)
    modelo=KNeighborsClassifier()
    modelos.append(modelo)
    modelo=LogisticRegression(C=1000.0,penalty='l2',max_iter=500)
    modelos.append(modelo)

    #Entrenamos todos los modelos con cross validation y elegimos el mejor
    best_model=None
    best_mean_score=-1

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for m in modelos:
            if verbose:
                print("Entrenando modelo",m,":")
            m.fit(X_train, y_train)

            scores = cross_val_score(modelo, X, y, cv=5)
            score_media=scores.mean()
            if verbose:
                print("Scores obtenidos:",scores)
                print("Score media:",score_media)
                print()
            if score_media>best_mean_score:
                best_mean_score=score_media
                best_model=m
    if verbose:
        print("\nEntrenamiento: OK")
        print("Mejor modelo:",best_model)
        print("Score:",best_mean_score)

    return best_model
    
        
'''*********************************************************ENTRENAMIENTO*****************************************************'''
        
#Generamos el dataframe resumen
if verbose:
    print("Generando dataframe de entrenamiento...")
df_train_test=libraries.genera_df_train(df)
if verbose:
    print("Listo\n")

#Porcentaje de los datos totales utilizado para entrenar el modelo
porcentaje_train=0.75

#Entre
modelo=train_model_normal(porcentaje_train,df_train_test)

#Guardamos el modelo resultante en un pickle
with open('modelo.pkl', 'wb') as file:
    pickle.dump(modelo, file)