import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from collections import defaultdict
from scipy.interpolate import UnivariateSpline
import math
import itertools

def filtro_acelerometro(datos):
    #
    # datos:    dataframe con los datos de tiempo (time) y acelerómetro (x, y, z)
    #
    
    # Copia de datos originales
    datos1 = datos.copy(deep = True)
    
    if np.mean(datos.ay) > 0:
        datos1.ay = -datos1.ay
    
    ind_x = np.where((datos1.ax >= np.mean(datos1.ax) + 8*np.std(datos1.ax)) | (datos1.ax <= np.mean(datos1.ax) - 8*np.std(datos1.ax)))
    ind_y = np.where((datos1.ay >= np.mean(datos1.ay) + 8*np.std(datos1.ay)) | (datos1.ay <= np.mean(datos1.ay) - 8*np.std(datos1.ay)))
    ind_z = np.where((datos1.az >= np.mean(datos1.az) + 8*np.std(datos1.az)) | (datos1.az <= np.mean(datos1.az) - 8*np.std(datos1.az)))
    
    if len(ind_x[0]) > 0:
        datos1.ax.iloc[ind_x] = np.nan
        datos1.ax = datos1.ax.interpolate(method ='linear', limit_direction ='both')
    if len(ind_y[0]) > 0:
        datos1.ay.iloc[ind_y] = np.nan
        datos1.ay = datos1.ay.interpolate(method ='linear', limit_direction ='both')
    if len(ind_z[0]) > 0:
        datos1.az.iloc[ind_z] = np.nan
        datos1.az = datos1.az.interpolate(method ='linear', limit_direction ='both')
    
    return datos1
    
def fases_marcha_individual(datos):
    #
    # datos:    dataframe con los datos de tiempo (time) y acelerómetro (x, y, z)
    #
    
    # datos_temp = datos1[['timestamp','ax','ay','az']].iloc[ind_temp].copy()
    # datos1 = datos_temp.copy(deep = True)
    
    # Copia de datos originales
    datos1 = datos.copy(deep = True)
        
    # Transformación a media cero las medidas
    datos1['ax'] = datos1['ax'] - np.mean(datos1['ax'])
    datos1['ay'] = datos1['ay'] - np.mean(datos1['ay'])
    datos1['az'] = datos1['az'] - np.mean(datos1['az'])
    
    #
    # Identificación de fases 1/3 vs 2/4
    #
    
    # Datos eje Y suavizados con 30 grados de libertad ¿?
    vector_temp = datos1['ay'].copy()
    ind_no_nas = np.where(-np.isnan(vector_temp))[0]
    vector_temp[np.isnan(vector_temp)] = 0
    ss = UnivariateSpline(ind_no_nas, vector_temp.iloc[ind_no_nas], k = 3, s = 1.75)
    datos1['ay_ss'] = np.nan
    datos1['ay_ss'].iloc[np.where(-np.isnan(datos1['ay']))] = ss(ind_no_nas)
    
    # Identificación de fases en función de la variable ay_ss
    signos_temp = np.sign(datos1['ay_ss'] + np.percentile(datos1['ay_ss'], 60))
    ind_signo = np.asarray(signos_temp.iloc[1:]) - np.asarray(signos_temp.iloc[:-1])
    ind_signo_pos = np.where(ind_signo != 0)[0] + 1
    fases = pd.DataFrame()
    fases['inicio'] = np.asarray(ind_signo_pos[:-1])
    fases['fin'] = np.asarray(ind_signo_pos[1:])
    fases['signo'] = np.sign(ind_signo[ind_signo != 0])[:-1]
    
    faaases=ss
    
    if fases.shape[0] == 0:
        return np.empty(datos.shape[0])
    
    #
    # Identificación de fases 1/2 vs 3/4
    #
    
    # Análisis a partie del eje X
    fases['lado'] = np.nan
    fases['valor'] = np.nan
    for i in range(0,fases.shape[0]):
        if fases['signo'].iloc[i] == 1:
            media_temp = round(np.mean([fases['inicio'].iloc[i], fases['fin'].iloc[i]]))
            ind_temp2 = list(set(range(0,datos.shape[0])) & set(range(-10 + media_temp,11 + media_temp)))
            fases['valor'].iloc[i] = np.mean(datos1['ax'].iloc[ind_temp2])
        
    if not all(np.isnan(fases['valor'])):
        ind_max = np.argmax(abs(fases['valor']))
        sign_temp = np.sign(fases['valor'].iloc[ind_max])
        fases['lado'] = np.append(np.resize([-sign_temp,-sign_temp,sign_temp,sign_temp], ind_max)[::-1],
                                  np.resize([sign_temp,sign_temp,-sign_temp,-sign_temp], fases.shape[0] - ind_max)[::-1])
    
    #
    # Asociación
    #
    
    # Identificación de estado en cada fase
    fases['estado'] = np.nan
    fases['estado'].loc[(fases['signo'] == 1)*(fases['lado'] == 1)] = 1
    fases['estado'].loc[(fases['signo'] == -1)*(fases['lado'] == 1)] = 2
    fases['estado'].loc[(fases['signo'] == 1)*(fases['lado'] == -1)] = 3
    fases['estado'].loc[(fases['signo'] == -1)*(fases['lado'] == -1)] = 4
    
    # Asociación de estados a los datos
    datos1['estado'] = np.nan
    for i in range(0,fases.shape[0]):
        datos1['estado'].iloc[range(fases['inicio'].iloc[i],fases['fin'].iloc[i] + 1)] = fases['estado'].iloc[i]
    
    return datos1['estado'],datos1['ay_ss'],faaases

def split(x, f):
    res = defaultdict(list)
    for v, k in zip(x, f):
        res[k].append(v)
    return res

def fases_marcha_global(datos):
    #
    # datos:    dataframe con los datos de tiempo (time) y acelerómetro (x, y, z)
    #
    
    # Copia de datos originales
    datos1 = datos.copy(deep = True)
    
    # Identificación de tramos
    ind_cambio = np.append(0, np.where(np.asarray(datos1['caminar'].iloc[1:]) -
                                       np.asarray(datos1['caminar'].iloc[:-1]) != 0)[0] + 1)
    
    # datos.to_csv('test.csv')
    
    if datos1['caminar'].iloc[0] == 0:
        ind_cambio = ind_cambio[1:]
    
    if (len(ind_cambio) % 2) == 1:
        ind_cambio = np.append(ind_cambio, ind_cambio[0])
    
    tramos = ind_cambio.copy().reshape(-1,2)
    tramos[:,1] = tramos[:,1] - 1 
    
    # Rastreo por tramo
    datos1['estado1'] = np.nan
    datos1['estado2'] = np.nan
    
    for i in range(0,tramos.shape[0]):
        # Filtro de outliers por tramo
        datos1.iloc[tramos[i,0]:(tramos[i,1] + 1)] = filtro_acelerometro(datos1.iloc[tramos[i,0]:(tramos[i,1] + 1)])
        
        # Partición y rastreo
        ind_part = max(1,round((tramos[i,1] - tramos[i,0])/700))
        vec_part = [j % ind_part for j in range(tramos[i,0],(tramos[i,1] + 1))]
        vec_part.sort()
        particiones = split(range(tramos[i,0],(tramos[i,1] + 1)), vec_part)
        
        for j in range(0,len(particiones)):
            if j == 0:
                ind_temp = range(particiones[j][0], particiones[j][-1] + 50)
            elif j == len(particiones):
                ind_temp = range(particiones[j][0] - 50, particiones[j][-1])
            else:
                ind_temp = range(particiones[j][0] - 50, particiones[j][-1] + 50)
            
            ind_temp = list(set(ind_temp) & set(range(0,datos1.shape[0])))
            est_temp,suav,faaases = fases_marcha_individual(datos1[['ax','ay','az']].iloc[ind_temp])
            if j % 2 == 1:
                datos1['estado1'].iloc[ind_temp] = est_temp
            else:
                datos1['estado2'].iloc[ind_temp] = est_temp
    
    # Unificación de estados
    datos1['estado'] = np.nan
    ind_eq = np.where(np.logical_or(datos1['estado1'] == datos1['estado2'], 
                      np.logical_and(np.isnan(datos1['estado1']), np.isnan(datos1['estado2']))))[0]
    datos1['estado'].iloc[ind_eq] = datos1['estado1'].iloc[ind_eq]
    
    ind_1 = np.where(np.logical_and(-np.isnan(datos1['estado1']), np.isnan(datos1['estado2'])))[0]
    datos1['estado'].iloc[ind_1] = datos1['estado1'].iloc[ind_1]
    
    ind_2 = np.where(np.logical_and(np.isnan(datos1['estado1']), -np.isnan(datos1['estado2'])))[0]
    datos1['estado'].iloc[ind_2] = datos1['estado2'].iloc[ind_2]
    
    ind_dif = np.where(np.logical_and(np.logical_and(-np.isnan(datos1['estado1']), 
                                                     -np.isnan(datos1['estado2'])), 
                                      datos1['estado1'] != datos1['estado2']))[0]
    datos1['estado'].iloc[ind_dif] = np.nan
    
    # Completitud de esados si el tramo sin datos es menor de 10 valores y tiene coherencia en la transición de estados
    if 1==1:
        ind_na = np.where(-np.isnan(datos1['estado']))[0]
        ind_na2 = np.where(np.logical_and(ind_na[1:] - ind_na[:-1] <= 10, ind_na[1:] - ind_na[:-1] > 1))[0]
        if len(ind_na2) > 0:
            for i in range(0,len(ind_na2)):
                ind_temp = range(ind_na[ind_na2[i]]+1,ind_na[ind_na2[i]+1])
                if (datos1['estado'].iloc[ind_na[ind_na2[i]+1]] - datos1['estado'].iloc[ind_na[ind_na2[i]]]) in [1,-3]:
                    if len(ind_temp) == 1: # i = 0
                        datos1['estado'].iloc[ind_temp] = datos1['estado'].iloc[ind_na[ind_na2[i]]].copy()
                    elif np.logical_and(len(ind_temp) % 2 == 1, len(ind_temp) != 1): # i = 10
                        datos1['estado'].iloc[ind_temp[:math.ceil(len(ind_temp)/2)]] = datos1['estado'].iloc[ind_na[ind_na2[i]]].copy()
                        datos1['estado'].iloc[ind_temp[math.ceil(len(ind_temp)/2):]] = datos1['estado'].iloc[ind_na[ind_na2[i] + 1]].copy()
                    elif len(ind_temp) % 2 == 0: # i = 3
                        datos1['estado'].iloc[ind_temp[:int(len(ind_temp)/2)]] = datos1['estado'].iloc[ind_na[ind_na2[i]]].copy()
                        datos1['estado'].iloc[ind_temp[int(len(ind_temp)/2):]] = datos1['estado'].iloc[ind_na[ind_na2[i] + 1]].copy()
    
    # Eliminación de tramos de marcha inferiores a 2 segundos
    if 1==2:
        ind_na = np.where(np.isnan(datos1['estado']))[0]
        ind_na2 = np.where(np.logical_and(ind_na[1:] - ind_na[:-1] <= 200, ind_na[1:] - ind_na[:-1] > 1))[0]
        if len(ind_na2) > 0:
            for i in range(0,len(ind_na2)):
                ind_temp = range(ind_na[ind_na2[i]]+1,ind_na[ind_na2[i]+1])
                datos1['estado'].iloc[ind_temp] = np.nan
    
    return datos1['estado']
