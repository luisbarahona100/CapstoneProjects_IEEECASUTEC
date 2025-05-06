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
import serial.tools.list_ports #pip install pyserial
import numpy as np

import csv 
import datetime
# VARIABLES GLOBALES
fs = 250                                     # Frecuencia de muestreo de Wang                                                     
freqsEstimul = [8, 10, 12, 14]               # Frecuencias de estimulación
freqReal = 8                                 # Frecuencia real usada para verificación
corrMin = 0.5                                # Umbral mínimo de correlación aceptable
N = int(6*fs)                                #Adquirir 6 segundos o 1500 muestras

def ConfiguarVerificarPuerto():
    """
    Verifica si el puerto '/dev/ttyACM0' está disponible y crea el objeto serial.
    Retorna el objeto serial si se puede abrir, o None si no está disponible.
    """
    port = 'COM5'  # Ajustar si estás en Windows, ejemplo: 'COM3'  ; linux: '/dev/ttyACM0'
    baudrate = 115200

    # Verificar si el puerto está en la lista de puertos disponibles
    available_ports = [p.device for p in serial.tools.list_ports.comports()]
    
    if port not in available_ports:
        print(f">> ❌ Error: El puerto {port} no está disponible.")
        print(f">> 🔎 Puertos disponibles: {available_ports}")
        return None

    try:
        serialObj = serial.Serial(port, baudrate, timeout=1)
        print(f">> ✅ Puerto {port} abierto correctamente.")
        return serialObj
    except serial.SerialException as e:
        print(f">> ❌ Error al abrir el puerto {port}: {e}")
        return None


def read_serial_data(serialObj, num_samples): #OK
    """Lee datos del puerto serial y devuelve un arreglo con las muestras."""
    buffer = []
    while len(buffer) < num_samples:
        if serialObj.in_waiting > 0:
            try:
                sample = float(serialObj.readline().decode().strip())
                buffer.append(sample)
            except ValueError:
                pass  # Ignorar líneas mal formateadas
    return np.array(buffer)

def load_eeg_data_online(serialObj, fs, N, nombre_archivo="EEG_data.csv"): #OK
    """
    Lee datos EEG desde el puerto serial, recorta las partes no útiles
    (inicio, delay y final del estímulo visual) y guarda los datos en un archivo CSV.
    """
    eeg_data=[]
    try:
        data = read_serial_data(serialObj, N)

        if len(data) > 250:  # Verifica que hay suficientes muestras
            t_ini = int(0.5 * fs)          # 0.5 s de transición al inicio
            delay = int(0.14 * fs)         # 140 ms de delay de estimulación
            t_fin = int(0.5 * fs)          # 0.5 s de transición al final

            corteInicio = t_ini + delay + 1
            eeg_data = data[corteInicio:-t_fin]  # Recorte útil

            # Guardar los datos recortados en un archivo CSV
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = "DataEEG"
            os.makedirs(folder, exist_ok=True)
            ruta_csv = os.path.join(folder, f"{timestamp}_{nombre_archivo}")

            np.savetxt(ruta_csv, eeg_data, delimiter=",", fmt="%.6f")
            print(f">> ✅ Datos EEG guardados en: {ruta_csv}")

            return np.array(eeg_data)

        else:
            print(f">> ⚠️ Advertencia: Se recibieron solo {len(data)} muestras. Conexión o sincronización fallida.")
            return np.array([])

    except Exception as e:
        print(f">> ❌ Error durante la adquisición EEG: {e}")
        return np.array([])


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
    #dataFiltrada = np.array([notch_filter(bandpass_filter(x, fs), fs) for x in dataCargada]) USAR PARA VARIOS CANALES
    dataFiltrada = notch_filter(bandpass_filter(dataCargada, fs), fs)

    return dataFiltrada

def detectarSSVEPconCCA(dataFiltrada, freqsEstimul, fs, corrMin): #OK
    """Detecta la frecuencia de SSVEP usando CCA."""
    X_N = dataFiltrada
    M = X_N.shape                           
    print(">> Forma de X_N: ", X_N.shape)               #(1 , 1214) tamaño esperado
    # Generar señales de referencia y asegurar que sea un array
    Y_K = np.array(generate_reference_signals(freqsEstimul, fs, M=len(X_N)), dtype=object)
    print(">> Forma de Y_K:", Y_K.shape, "elementos")   #(4 , 8, 1214) tamaño esperado

    cca = CCA(n_components=1)
    max_corr = 0
    detected_freq = None

    for i, Y in enumerate(Y_K):
        Y = np.array(Y)  # Asegurar que Y sea un array de NumPy
        
        print(f">> Comparando con frecuencia {freqsEstimul[i]} Hz, forma de Y:", Y.shape)

        if X_N.shape[0] != Y.shape[1]:
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

def prueba1_online (N, fs, corrMin, freqsEstimul, freqReal):
    #INPUT DATA EEG OFFLINE DE LA BASE DE DATOS DE WANG 1x1500 muestras a fs=250Hz , 6 segundos
    #ADQUIERE, DETECTA, CLASIFICA, EJECUTA UN COMANDO 
    print("Prueba N° 1 - Detección de SSVEP usanco CCA en PC\n")
    print("Paso 1 - Configurar Puerto Serial, verificar su disponibilidad y generar Objeto serial\n")
    serialObj = ConfiguarVerificarPuerto()
    print(f">> SerialObjt:{serialObj} \n")

    print("Paso 2 - Cargar data EEG del puerto a memoria de programa y recortarla\n")
    dataCargada = load_eeg_data_online(serialObj, fs, N) 
    print(f">> Dimensiones de la data cargada y recortada: {dataCargada.shape}\n")

    print ("Paso 3 - Filtrar la data EEG<\n")
    dataFiltrada = filtrarData(dataCargada, fs) 
    print(f">> Dimensiones de la data filtrada: {dataFiltrada.shape}\n")

    print("Paso 4 - Detectar SSVEP usando CCA \n ")
    maxCorr, freqDetectada = detectarSSVEPconCCA(dataFiltrada, freqsEstimul, fs, corrMin)
    print(f">> Frecuencia detectada: {freqDetectada} vs Frecuencia real: {freqReal} con una max_corr: {maxCorr}\n")

    print("Paso 5 - Generar comandos en base a la frecuencia detectada")
    comando = generarComandosDiscretos(freqDetectada)
    print(f">> Comando generado: {comando}")


#PRUEBAS DE RENDIMIENTO
def medirTiempoEjecusion(): #FUNCIONA -> 13.36seg
    print("MEDIR TIEMPO DE EJECUCIÓN DE PRUEBA1_OFFLINE\n")
    start = time.time()
    #INICIO_FUNCIÓN A TESTEAR
    prueba1_online (N, fs, corrMin, freqsEstimul, freqReal) #OK
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
    prueba1_online (N, fs, corrMin, freqsEstimul, freqReal)
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
    prueba1_online (N, fs, corrMin, freqsEstimul, freqReal)
    #FIN_F
    energy_meter.stop(tag='CCA')

    # Obtener datos
    handler = CSVHandler('resultados_energia.csv')
    handler.process(energy_meter.get_trace())

    # Si quieres imprimirlo directamente:
    
    df = pd.read_csv('resultados_energia.csv')
    print(df)

# D. PROBAR CADA FUNCIÓN
prueba1_online (N, fs, corrMin, freqsEstimul, freqReal) #OK


# E. TESTEAR RENDIMIENTO EN TIEMPO
#medirTiempoEjecusion()  #OK 13.36mseg

# F. TESTEAR  CONSUMO DE MEMORIA
#prueba1_offline(nameFile, fs, corrMin, freqsEstimul, freqReal) #OK
#Ejecuar el code con : python -m memory_profiler code_cca_pc.py
#Esto te dará un reporte línea por línea del uso de memoria en MB de la función que tenga el @profile

# G. TESTEAR CONSUMO DE CPU Y USO GENERAL DEL SISTEMA
#medirConsumoCPU()

# H. TESTEAR CONSUMO DE ENERGÍA
