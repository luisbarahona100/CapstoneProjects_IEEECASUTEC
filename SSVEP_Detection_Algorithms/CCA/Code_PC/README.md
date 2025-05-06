# 🧠 Detección de SSVEP en Tiempo Real y Offline usando CCA

Este proyecto implementa un sistema para detectar **Potenciales Evocados Visuales Estables (SSVEP)** a partir de señales EEG, utilizando el algoritmo **Canonical Correlation Analysis (CCA)**. Se puede aplicar tanto para señales en tiempo real como para análisis offline con datos pregrabados.

---

## 📌 Objetivo

Detectar frecuencias de estimulación visual (SSVEP) presentes en señales EEG usando el algoritmo CCA, con datos adquiridos:

- En **tiempo real** (a través de serial)
- Desde archivos **offline** (.csv)

---

## 🧰 Librerías Usadas

| Librería              | Propósito                                              |
|-----------------------|--------------------------------------------------------|
| `matplotlib`, `numpy`, `pandas` | Visualización y manipulación de datos       |
| `scipy.signal`        | Filtrado de señales (Butterworth, Notch)              |
| `sklearn.cross_decomposition.CCA` | Implementación del algoritmo CCA         |
| `scipy.io.loadmat`    | Carga de archivos `.mat`                               |
| `serial`              | Comunicación con dispositivos externos vía serial      |
| `pyJoules`, `psutil`  | Perfilado energético y de memoria                      |

---

## 📁 Estructura de Carpetas

├── code_cca_pc.py
├── DataEEG/
│ └── Xn_10Hz_Oz_B1_S1.csv



---

## ⚙️ Configuración Inicial

- **Frecuencia de muestreo**: `fs = 250 Hz`
- **Frecuencias de estimulación**: `[8, 10, 12, 14] Hz`
- **Frecuencia real esperada**: `10 Hz`
- **Correlación mínima aceptada**: `0.5`
- **Archivo CSV de entrada**: `"Xn_10Hz_Oz_B1_S1.csv"` ubicado en `/DataEEG`

---

## 🔄 Flujo del Programa

1. **Verificación de archivo** (`verificarRuta`)
2. **Carga de datos EEG** (`load_eeg_data_offline`)
3. **Filtrado**:
   - Pasa banda 7-70Hz (`bandpass_filter`)
   - Notch a 50Hz (`notch_filter`)
4. **Generación de señales de referencia** (`generate_reference_signals`)
5. **Aplicación del algoritmo CCA** (`detectarSSVEPconCCA`)
6. **Comando discreto de salida según frecuencia detectada** (`generarComandosDiscretos`)

---

## 🔬 CCA y Señales de Referencia

Se generan señales seno y coseno para cada frecuencia objetivo hasta la 4ta armonía. Se comparan con la señal EEG real para encontrar la frecuencia con mayor correlación.

```python
ref = [
    sin(2πft), cos(2πft), 
    sin(4πft), cos(4πft), 
    ...
]
```

## ✅ Resultado Esperado

Al ejecutar el sistema correctamente, se mostrará en consola:

- 📂 **Ruta del archivo EEG cargado**
- 📐 **Dimensiones** del archivo (muestras × canales)
- 📊 **Frecuencia detectada** con mayor correlación
- 🎮 **Comando resultante** basado en la frecuencia detectada  
  _(Ejemplo: “🔽 Abajo”, “➡️ Derecha”)_

---

## 🎮 Comandos Asociados por Frecuencia

| ⚡ Frecuencia Detectada | 🧭 Comando Asociado |
|------------------------|---------------------|
| ~8 Hz                  | ⬆️ Arriba           |
| ~10 Hz                 | ⬇️ Abajo            |
| ~12 Hz                 | ➡️ Derecha          |
| ~14 Hz                 | ⬅️ Izquierda        |

---

## 🧪 Cómo Ejecutar el Script

```bash
# Ejecución estándar
python code_cca_pc.py

# Ejecución con perfilado de memoria
python -m memory_profiler code_cca_pc.py
```

## 👨‍💻 Autor
Luis David Barahona Valdivieso  
Estudiante e Investigador de Ingeniería Electrónica  
Universidad de Ingeniería y Tecnología - UTEC  
