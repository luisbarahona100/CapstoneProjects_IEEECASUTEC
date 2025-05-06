"""
OBJETIVO: Detectar SSVEP de una data obtenida en tiempo real y de una offline usando el algoritmo CCA. 
"""
#A. LIBRERÍAS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.cross_decomposition import CCA ##pip install scikit-learn
from scipy.io import loadmat #para manipular data .mat
import os

import time 
from memory_profiler import profile
import psutil
from pyJoules.energy_meter import EnergyMeter
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.handler.csv_handler import CSVHandler
from io import StringIO

import serial 
#B. VARIABLES GLOBALES
fs = 250                                     #frecuencia de muestreo de Wang                                                     
freqsEstimul = [8, 10, 12, 14]               #frecuencias de estimulación
freqReal = 10                                #tiene que ser seteada en cada testeo. Se usa solo para verificar que la frecuencia detectada, sea la misma.
corrMin = 0.5                                #si el ensayo genera una corr menor, entonces el ensayo es fallido
nameFile = "Xn_10Hz_Oz_B1_S1.csv" 

#C. FUNCIONES
def verificarRuta(nameFile): #OK Si code.py está al mismo nivel que DataEEG
    # GOAL: Verificar que la ruta entregada se pueda abrir exitosamente.

    ruta_relativa = os.path.join(".", "DataEEG", nameFile)
    ruta_absoluta = os.path.abspath(ruta_relativa)

    print(f">> 🔍 Probando acceso a: {ruta_absoluta}\n")

    if not os.path.exists(ruta_absoluta):
        print(">> ❌ El archivo no existe.")
        return None

    try:
        with open(ruta_absoluta, 'rb') as f:
            print(">> ✅ El archivo se puede abrir correctamente.")
    except PermissionError:
        print(">> ❌ No tienes permiso para leer el archivo.")
        return None
    except Exception as e:
        print(f">> ⚠️ Otro error ocurrió: {e}")
        return None

    return ruta_absoluta

def load_eeg_data_offline(rutaFile): #OK
    # load_eeg_data() corta 0.5s al inicio + 0.14 segundos de delay y 0.5seg al final -> 1215 muestras útiles con info de estimulación
    
    rutaFile = [rutaFile]
    eeg_data = []
    
    for f in rutaFile:
        try:
            data = pd.read_csv(f, header=None).values.flatten()  # Leer y aplanar
            if len(data) > 250:  # Asegurar que haya suficientes muestras
                transicionInicioEnsayo = int(0.5*fs) #info no útil antes de la estimulación visual
                transicionFinEnsayo=int(0.5*fs) #info no útil después de la estimulación visual
                delayStimulus = int(0.14*fs) #info de delay de 140ms según wang durante la estimulación visual
                corteInicio = transicionInicioEnsayo+delayStimulus + 1
                data = data[corteInicio:-transicionFinEnsayo]  # Quitar primeras y últimas 125 muestras (0.5seg y 0.5 seg)
                
            else:
                print(f">> Advertencia: {f} tiene menos de 250 muestras y no se procesará.")
                continue
            eeg_data.append(data)
        except FileNotFoundError:
            print(f">> Error: No se encontró el archivo {f}")
        except Exception as e:
            print(f">> Error al procesar {f}: {e}")
    
    #print("Forma de la matriz eeg_data:", eeg_data)
    return np.array(eeg_data) #size of eeg_data recortada:  (1, 1214)

def bandpass_filter(data, fs, lowcut=7, highcut=70, order=4):
    """Aplica un filtro pasa banda (7-70Hz)."""
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, fs, notch_freq=50, quality_factor=30):
    """Aplica un filtro notch para eliminar interferencia a 50Hz."""
    b, a = iirnotch(notch_freq, quality_factor, fs)
    return filtfilt(b, a, data)

def generate_reference_signals(freqs, fs, M):
    """Genera señales seno y coseno para las frecuencias dadas."""
    t = np.arange(M) / fs
    reference_signals = []
    for f in freqs:
        ref = np.vstack([np.sin(1*2* np.pi * f * t), np.cos(1*2* np.pi * f * t),
                         np.sin(2*2* np.pi * f * t), np.cos(2*2* np.pi * f * t),
                         np.sin(3*2* np.pi * f * t), np.cos(3*2* np.pi * f * t),
                         np.sin(4*2* np.pi * f * t), np.cos(4*2* np.pi * f * t)])
        reference_signals.append(ref)
    return reference_signals

def filtrarData(dataCargada, fs): #OK
    dataFiltrada = np.array([notch_filter(bandpass_filter(x, fs), fs) for x in dataCargada])
    return dataFiltrada

def detectarSSVEPconCCA(dataFiltrada, freqsEstimul, fs, corrMin): #OK
    """Detecta la frecuencia de SSVEP usando CCA."""
    X_N = dataFiltrada
    M = X_N.shape                           
    print(">> Forma de X_N: ", X_N.shape)               #(1 , 1214) tamaño esperado
    # Generar señales de referencia y asegurar que sea un array
    Y_K = np.array(generate_reference_signals(freqsEstimul, fs, M[1]), dtype=object)
    print(">> Forma de Y_K:", Y_K.shape, "elementos")   #(4 , 8, 1214) tamaño esperado

    cca = CCA(n_components=1)
    max_corr = 0
    detected_freq = None

    for i, Y in enumerate(Y_K):
        Y = np.array(Y)  # Asegurar que Y sea un array de NumPy
        
        print(f">> Comparando con frecuencia {freqsEstimul[i]} Hz, forma de Y:", Y.shape)

        if X_N.shape[1] != Y.shape[1]:
            print(f">> ❌ Error: X_N tiene {X_N.shape[1]} muestras y Y[{i}] tiene {Y.shape[1]}.")
            continue  # Saltar esta iteración si hay un problema de tamaño

        try:
            cca.fit(X_N.T, Y.T)
            X_c, Y_c = cca.transform(X_N.T, Y.T)
            rho = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]

            if rho > max_corr:
                max_corr = rho
                detected_freq = freqsEstimul[i]

        except Exception as e:
            print(f">> ⚠️ Error en CCA con frecuencia {freqsEstimul[i]} Hz:", str(e))

    if max_corr < corrMin or detected_freq != freqReal:
        print(f">> ❌ Ensayo fallido: corr={max_corr:.2f}, freq_detectada={detected_freq}, freq_real={freqReal}")
        detected_freq = 0
    else:
        print(f">> ✅ Ensayo exitoso: corr={max_corr:.2f}, freq_detectada={detected_freq} Hz, freq_real={freqReal}")

    return max_corr, detected_freq

def generarComandosDiscretos(frecuencia):
    #UNA MEJORA ES: Tener al accuracy como input, de modo tal que se generen comandos solo si el accuracy es mayor a 70% por ejemplo.
    """Envía un comando por serial según la frecuencia detectada."""
    comandos = {
        (7.5, 8.5): "Arriba",
        (9.5, 10.5): "Abajo",
        (11.5, 12.5): "Derecha",
        (13.5, 14.5): "Izquierda"
    }
    
    for (low, high), comando in comandos.items():
        if low <= frecuencia <= high:
            print(f"Comando: {comando}")
            #ser = serial.Serial('/dev/ttyUSB0', 9600)  # Ajusta el puerto según el dispositivo
            #ser.write(comando.encode())
            #ser.close()
            return comando
    
    return "Sin comando"

#PRUEBAS   DE FUNCIONALIDAD
#@profile usar solo cuando ejecutes:  python -m memory_profiler code_cca_pc.py
def prueba1_offline (nameFile, fs, corrMin, freqsEstimul, freqReal):
    #INPUT DATA EEG OFFLINE DE LA BASE DE DATOS DE WANG 1x1500 muestras a fs=250Hz , 6 segundos
    #ADQUIERE, DETECTA, CLASIFICA, EJECUTA UN COMANDO 
    print("Prueba N° 1 - Detección de SSVEP usanco CCA en PC\n")
    print("Paso 1 - Verificar que se pueda acceder al archivo entregado y generar ruta\n")
    rutaFile = verificarRuta(nameFile)
    print(f">> Ruta:{rutaFile} \n")

    print("Paso 2 - Cargar data EEG de carpeta a memoria de programa y recortarla\n")
    dataCargada = load_eeg_data_offline(rutaFile) 
    print(f">> Dimensiones de la data cargada y recortada: {dataCargada.shape}\n")  #(1, 1214)

    print ("Paso 3 - Filtrar la data EEG<\n")
    dataFiltrada = filtrarData(dataCargada, fs) 
    print(f">> Dimensiones de la data filtrada: {dataFiltrada.shape}\n")            #(1, 1214)

    print("Paso 4 - Detectar SSVEP usando CCA \n ")
    maxCorr, freqDetectada = detectarSSVEPconCCA(dataFiltrada, freqsEstimul, fs, corrMin)  #Y_K -> (4, 8, 1214)
    print(f">> Frecuencia detectada: {freqDetectada} vs Frecuencia real: {freqReal} con una max_corr: {maxCorr}\n")

    print("Paso 5 - Generar comandos en base a la frecuencia detectada")
    comando = generarComandosDiscretos(freqDetectada)
    print(f">> Comando generado: {comando}")

#PRUEBAS DE RENDIMIENTO
def medirTiempoEjecusion(): #FUNCIONA -> 13.36seg
    print("MEDIR TIEMPO DE EJECUCIÓN DE PRUEBA1_OFFLINE\n")
    start = time.time()
    #INICIO_FUNCIÓN A TESTEAR
    prueba1_offline(nameFile, fs, corrMin, freqsEstimul, freqReal) #OK
    #INICIO_FUNCIÓN A TESTEAR
    end = time.time()
    print (f">> Tiempo de ejecución: {end - start:.6f} segundos")

def medirConsumoCPU(): #FUNCIONA -> 150MB -> 154MB
    print("\n================== MEDIR CONSUMO DE CPU DE LA FUNCIÓN prueba1_offline()=====================\n")
    process = psutil.Process(os.getpid())

    print(f">> Memoria inicial: {process.memory_info().rss / 1024 ** 2:.2f} MB") #GENERA -> 150.44MB

    start = time.time()
    #INICIO_FUNCIÓN A TESTEAR
    print("\n======================INICIO EJECUCIÓN DE PRUEBA1_OFFFLINE()===================================\n")
    prueba1_offline(nameFile, fs, corrMin, freqsEstimul, freqReal)
    print("\n======================FIN EJECUCIÓN DE PRUEBA1_OFFFLINE()=================================\n")
    #FIN_FUNCIÓN A TESTEAR
    end = time.time()

    print(f">> Memoria final: {process.memory_info().rss / 1024 ** 2:.2f} MB")  #GENERA -> 153.99MB
    print(f">> Tiempo: {end - start:.6f} s")

def medirEnergia(): #NO FUNCIONA
    # Configurar medidor de energía
    energy_meter = EnergyMeter(domains=[RaplPackageDomain(0)])

    # Iniciar medición
    energy_meter.start(tag='CCA')
    #INICIO_F
    prueba1_offline(nameFile, fs, corrMin, freqsEstimul, freqReal)
    #FIN_F
    energy_meter.stop(tag='CCA')

    # Obtener datos
    handler = CSVHandler('resultados_energia.csv')
    handler.process(energy_meter.get_trace())

    # Si quieres imprimirlo directamente:
    
    df = pd.read_csv('resultados_energia.csv')
    print(df)

# D. PROBAR CADA FUNCIÓN
prueba1_offline(nameFile, fs, corrMin, freqsEstimul, freqReal) #OK


# E. TESTEAR RENDIMIENTO EN TIEMPO
#medirTiempoEjecusion()  #OK 13.36mseg

# F. TESTEAR  CONSUMO DE MEMORIA
#prueba1_offline(nameFile, fs, corrMin, freqsEstimul, freqReal) #OK
#Ejecuar el code con : python -m memory_profiler code_cca_pc.py
#Esto te dará un reporte línea por línea del uso de memoria en MB de la función que tenga el @profile

# G. TESTEAR CONSUMO DE CPU Y USO GENERAL DEL SISTEMA
#medirConsumoCPU()

# H. TESTEAR CONSUMO DE ENERGÍA
