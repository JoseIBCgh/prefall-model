import libraries
import os
import sys

# Obtener los argumentos del script
argumentos = sys.argv[1:]

#Si no se pasan argumentos, no se eejcuta el script
if len(argumentos)!=3:
    print('Error: no se ha especificado el número de argumentos correcto. Especifique la ruta del fichero sobre el que generar la gráfica, el valor de comienzo de la gráfica y el el valor final en centésimas de segundo.\n'
         + 'Ejemplo: python3 Genera_grafica.py ./datos/CTIC_casos/1/20230503-11-43-14-971.txt 1000 1500')
    sys.exit(1) 

try:
    #Leemos el argumento
    ruta = argumentos[0]

    #Comprobamos si existe
    if os.path.exists(ruta):
        inicio=int(argumentos[1])
        fin=int(argumentos[2])
        if inicio<0 or fin<inicio:
            print("Error: Rango de valores no válido")
        else:
            #Generamos la gráfica
            libraries.genera_grafica_fases(ruta,[inicio,fin])
    else:
        print("Error: No existe el fichero especificado")
        sys.exit(1)
        
except Exception as e:
    print("Ha habido un error")
    print(e)
    sys.exit(1)
