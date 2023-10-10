#Imports básicos
import pandas as pd
import numpy as np
import libraries
import random
import os
import sys
import io
import warnings
import pickle

''' ARGUMENTOS PARA PERSONALIZAR EL SCRIPT '''
#Argumento para imprimir por consola o no los resultados. Por defecto No
verbose=False
#Argumento para almacenar el resultado de las predicciones del modelo en un fichero txt. Por defecto Sí
save_txt=True


''' ARGUMENTOS DEL SCRIPT '''
# Obtener los argumentos del script
argumentos = sys.argv[1:]

#Si no se pasan argumentos, no se eejcuta el script
if len(argumentos)==0:
    print('Error: no se ha especificado ningún argumento')
    sys.exit(1)

#Almacenamiento de ficheros
ficheros=[]    
 
try:
    #Si se le pasa solo un argumento    
    if len(argumentos)==1:

        #Leemos el argumento
        ruta = argumentos[0]

        #Si es un directorio
        if os.path.isdir(ruta):
            for archivo in os.listdir(ruta):
                ficheros.append(os.path.join(ruta, archivo))
        #Si es un nombre de fichero
        else:
            ficheros.append(ruta)

    #Si se le pasan varios argumentos (ficheros)        
    else:
        for a in argumentos:
            ficheros.append(a)
except Exception as e:
    print("Los datos pasado en el comando no son correctos")
    sys.exit(1)


''' PREPARACION DEL MODELO '''
modelo = pickle.load(open('modelo.pkl', 'rb'))

''' PREPARACION FICHERO SALIDA SI ES NECESARIO '''
if save_txt:
    prefix = "resultado"
    extension = ".txt"

    # Obtener la lista de archivos en el directorio
    directorio='./output'
    files = os.listdir(directorio)

    # Encontrar el archivo más reciente y su número
    latest_file_num = 0
    for file in files:
        if file.startswith(prefix) and file.endswith(extension):
            file_num = int(file[len(prefix):-len(extension)])
            if file_num > latest_file_num:
                latest_file_num = file_num

    # Crear un nuevo archivo con el siguiente número
    next_file_num = latest_file_num + 1
    next_file_name = prefix + str(next_file_num) + extension
    next_file_path = os.path.join(directorio, next_file_name)


    

''' OBTENEMOS LA PROBABILIDAD DE CAIDA DE CADA FICHERO '''
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    i=1
    with open(next_file_path, 'w') as file:
        
        for fichero in ficheros:

            # Abrir el archivo en modo lectura
            with open(fichero, 'r') as f:
                # Leer el contenido del archivo
                contenido = f.read()

            # Reemplazar los tabuladores por espacios en memoria
            contenido = contenido.replace('\t', ' ')

            # Crear un objeto DataFrame de Pandas a partir del contenido
            datos = pd.read_csv(io.StringIO(contenido), delim_whitespace=True)

            #Si el fichero está vacío lo pasamos
            if datos.empty:
                continue

            #Renombramos columnas para adaptarlo al script libraries
            datos.rename(columns={'ACC_X': 'ax', 'ACC_Y': 'ay', 'ACC_Z': 'az'}, inplace=True)
            datos.rename(columns={'ACC_X': 'ax', 'ACC_Y': 'ay', 'ACC_Z': 'az'}, inplace=True)
            datos.rename(columns={'ACC_X': 'ax', 'ACC_Y': 'ay', 'ACC_Z': 'az'}, inplace=True)

            # Filtro de outliers general. Saltamos los primeros segundos para eliminar datos anomalos
            datos = libraries.filtro_acelerometro(datos.iloc[500:-500, :])

            #Definimos que el sujeto está caminando durante todo el proceso de medición. *A futuro esto puede modificarse
            datos['caminar'] = [1 for i in range(len(datos))]

            #Obtenemos las fases de la marcha
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # Filtro de tramos caminados
                if any(datos['caminar'] == 1):
                    if any(datos['caminar'][-10:] == 1):
                        datos['caminar'].iloc[-10:] = 0
                    datos['estado']=libraries.fases_marcha_global(datos)

            # Agrupa los datos por fases
            #datos=datos.dropna()
            fases = datos.groupby('estado')

            # Itera sobre cada grupo de fases
            fila = {}
            for i, (fase, grupo) in enumerate(fases, start=1):
                # Calcula la duración de la fase
                duracion = grupo['TIME'].iloc[-1] - grupo['TIME'].iloc[0]
                duracion=len(grupo)/100

                #acelerometro
                ax_mean = grupo['ax'].mean()
                ay_mean = grupo['ay'].mean()
                az_mean = grupo['az'].mean()

                ax_std = grupo['ax'].std()
                ay_std = grupo['ay'].std()
                az_std = grupo['az'].std()

                #CALCULAMOS VALORES MEDIOS DE SENSORES PARA CADA FASE DE MARCHA
                #Aceleracion lineal
                lax_mean = grupo['LACC_X'].mean()
                lay_mean = grupo['LACC_Y'].mean()
                laz_mean = grupo['LACC_Z'].mean()

                #giroscopio
                gx_mean = grupo['GYR_X'].mean()
                gy_mean = grupo['GYR_Y'].mean()
                gz_mean = grupo['GYR_Z'].mean()

                #magnetometro
                mx_mean = grupo['MAG_X'].mean()
                my_mean = grupo['MAG_Y'].mean()
                mz_mean = grupo['MAG_Z'].mean()

                #quaterniones
                qx_mean = grupo['QUAT_X'].mean()
                qy_mean = grupo['QUAT_Y'].mean()
                qz_mean = grupo['QUAT_Z'].mean()
                qw_mean = grupo['QUAT_Z'].mean()

                #CALCULAMOS DESVIACION TIPICA DE SENSROES PARA CADA FASE DE MARCHA
                #Aceleracion lineal
                lax_std = grupo['LACC_X'].std()
                lay_std = grupo['LACC_Y'].std()
                laz_std = grupo['LACC_Z'].std()

                #giroscopio
                gx_std = grupo['GYR_X'].std()
                gy_std = grupo['GYR_Y'].std()
                gz_std = grupo['GYR_Z'].std()

                #magnetometro
                mx_std = grupo['MAG_X'].std()
                my_std = grupo['MAG_Y'].std()
                mz_std = grupo['MAG_Z'].std()

                #quaterniones
                qx_std = grupo['QUAT_X'].std()
                qy_std = grupo['QUAT_Y'].std()
                qz_std = grupo['QUAT_Z'].std()
                qw_std = grupo['QUAT_Z'].std()


                # Agrega las características al diccionario fila
                fila.update({
                    'duracion_f'+str(i): duracion,
                    'ax_mean_f'+str(i): ax_mean, 'ay_mean_f'+str(i): ay_mean, 'az_mean_f'+str(i): az_mean,
                    'ax_std_f'+str(i): ax_std, 'ay_std_f'+str(i): ay_std, 'az_std_f'+str(i): az_std,
                    'lax_mean_f'+str(i): lax_mean, 'lay_mean_f'+str(i): lay_mean, 'laz_mean_f'+str(i): laz_mean,
                    'lax_std_f'+str(i): lax_std, 'lay_std_f'+str(i): lay_std, 'laz_std_f'+str(i): laz_std,
                    'gx_mean_f'+str(i): gx_mean, 'gy_mean_f'+str(i): gy_mean, 'gz_mean_f'+str(i): gz_mean,
                    'gx_std_f'+str(i): gx_std, 'gy_std_f'+str(i): gy_std, 'gz_std_f'+str(i): gz_std,
                    'mx_mean_f'+str(i): mx_mean, 'my_mean_f'+str(i): my_mean, 'mz_mean_f'+str(i): mz_mean,
                    'mx_std_f'+str(i): mx_std, 'my_std_f'+str(i): my_std, 'mz_std_f'+str(i): mz_std,
                })

            # Convierte la fila en un DataFrame y añade la fila al dataframe final
            df_fila = pd.DataFrame([fila])
            

            #Predecimos la probabilidad de caida con el modelo
            prediccion=modelo.predict_proba([df_fila.iloc[0].tolist()])[0][1]
            texto="Probabildiad de caida de la medicion "+ str(i) + "con nombre de fichero " + str(fichero)+": "+str(prediccion)
            
            i=i+1
            
            if verbose:
                print(texto)
            if save_txt:
                file.write(texto+"\n")


