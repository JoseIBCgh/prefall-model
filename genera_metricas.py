import pandas as pd
import numpy as np
import libraries

import os
import io
import sys

import warnings

# Obtener los argumentos del script
argumentos = sys.argv[1:]

#Si no se pasan argumentos, no se eejcuta el script
if len(argumentos)!=1:
    print('Error: no se ha especificado el número de argumentos correcto. Especifique la ruta del fichero sobre el que generar la gráfica, el valor de comienzo de la gráfica y el el valor final en centésimas de segundo.\n'
         + 'Ejemplo: python3 Genera_grafica.py ./datos/CTIC_casos/1/20230503-11-43-14-971.txt 1000 1500')
    sys.exit(1) 

try:
    #Leemos el argumento
    ruta = argumentos[0]

    #Comprobamos si existe
    if os.path.exists(ruta):
        
        #Calculamos las métricas y generamos el csv
        try:
            # Abrir el archivo en modo lectura
            with open(ruta, 'r') as f:
                # Leer el contenido del archivo
                contenido = f.read()
        except Exception as e:
            print("Ha habido un error con el fichero:")
            print(e)
            exit(1)

        # Reemplazar los tabuladores por espacios en memoria
        contenido = contenido.replace('\t', ' ')

        # Crear un objeto DataFrame de Pandas a partir del contenido
        elementos = pd.read_csv(io.StringIO(contenido), delim_whitespace=True)
        elementos = elementos.reset_index()

        #Renombramos columnas para adaptarlo al script libraries
        elementos.rename(columns={'ACC_X': 'ax', 'ACC_Y': 'ay', 'ACC_Z': 'az'}, inplace=True)
        elementos.rename(columns={'ACC_X': 'ax', 'ACC_Y': 'ay', 'ACC_Z': 'az'}, inplace=True)
        elementos.rename(columns={'ACC_X': 'ax', 'ACC_Y': 'ay', 'ACC_Z': 'az'}, inplace=True)

        # Filtro de outliers general. Saltamos los primeros segundos para eliminar datos anomalos
        elementos = libraries.filtro_acelerometro(elementos.iloc[500:-500, :])

        #Definimos que el sujeto está caminando durante todo el proceso de medición. *A futuro esto puede modificarse
        elementos['caminar'] = [1 for i in range(len(elementos))]

        #Obtenemos las fases de la marcha
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Filtro de tramos caminados
            if any(elementos['caminar'] == 1):
                if any(elementos['caminar'][-10:] == 1):
                    elementos['caminar'].iloc[-10:] = 0
                elementos['estado']=libraries.fases_marcha_global(elementos)
            elementos['Paso'] = libraries.assign_steps(elementos['estado'].to_numpy())
            
            #Agrupamos por fase
            fases = elementos.groupby('estado')

            # Itera sobre cada grupo de fases
            fila = {}
            fila.update({
                    'duracion_total_medicion': elementos['TIME'].iloc[-1]+500, 'duracion_real_analizada': elementos['TIME'].iloc[-1]-500,
                    'n_pasos_totales': elementos['Paso'].iloc[-1], 'n_pasos_minuto': elementos['Paso'].iloc[-1]/(elementos['TIME'].iloc[-1]/6000),
                })
            for i, (fase, grupo) in enumerate(fases, start=1):
                # Calcula la duración de la fase
                duracion = len(grupo['TIME'])
                duracion_porcentaje=duracion/len(elementos.dropna(subset=['estado']))

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
                    'duracion_porcentaje_f'+str(i): duracion_porcentaje, 'duracion_total_f'+str(i): duracion,
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
            df_fila.to_csv("./output/metricas_"+ruta.split("/")[-1].split(".")[0]+".csv",index=False)    
                
    else:
        print("Error: No existe el fichero especificado")
        sys.exit(1)
        
except Exception as e:
    print("Ha habido un error")
    print(e)
    sys.exit(1)

