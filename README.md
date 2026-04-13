# Comparación del Desempeño de Modelos de Clasificación para la Predicción de la Diabetes como Enfermedad Crónica en Bucaramanga

> **Comparison of the Performance of Classification Models for the Prediction of Diabetes as a Chronic Disease in Bucaramanga**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-%23337AB7.svg?style=for-the-badge&logo=xgboost&logoColor=white)
![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)
![Datos Abiertos](https://img.shields.io/badge/Open%20Data-Colombia-green?style=for-the-badge)
![Estado](https://img.shields.io/badge/Estado-En%20Revisi%C3%B3n%20Editorial-orange?style=for-the-badge)

---

**Autor:** Ing. Juan Manuel Betancur  
**Afiliación:** Corporación Unificada Nacional de Colombia (CUN) — Bogotá, Colombia  
**Contacto:** Juan.betancurl@cun.edu.co  
**ORCID:** [0009-0000-7720-5918](https://orcid.org/0009-0000-7720-5918)

---

## 📋 Descripción del Proyecto

Este proyecto tiene como objetivo comparar el desempeño de distintos modelos de clasificación para la predicción de la **diabetes mellitus como enfermedad crónica** en el municipio de Bucaramanga, utilizando datos abiertos oficiales del sector salud colombiano. El propósito es aportar evidencia analítica que apoye estrategias de **detección temprana en salud pública**.

---

## 📊 Fuente de Datos

Los datos utilizados en esta investigación provienen del **Portal de Datos Abiertos del Estado Colombiano**, específicamente del conjunto de datos:

> **39. Enfermedades crónicas en el municipio de Bucaramanga**  
> Administrado por la Alcaldía de Bucaramanga  
> 🔗 [https://www.datos.gov.co/Salud-y-Protecci-n-Social/39-Enfermedades-cr-nicas-en-el-municipio-de-Bucara/4iz7-suhz/about_data](https://www.datos.gov.co/Salud-y-Protecci-n-Social/39-Enfermedades-cr-nicas-en-el-municipio-de-Bucara/4iz7-suhz/about_data)

El dataset cuenta con:
- **110.693 registros** clínicos y sociodemográficos
- **22 variables** (atributos clínicos, sociodemográficos y de comorbilidades)

---

## 🗂️ Estructura del Repositorio

```
├── Datasets/              # Datos crudos y procesados (CSV)
├── NoteBooks/             # Notebooks de Google Colab (.ipynb)
│   ├── Fase 1 - ETL y EDA
│   └── Fase 2 - Modelado y Evaluación
├── Modelos Guardados/     # Modelos serializados (.pkl)
├── Imágenes/              # Visualizaciones y figuras generadas
└── README.md
```

---

## ⚙️ Metodología

El desarrollo metodológico se estructuró en **dos fases secuenciales**, ejecutadas en **Google Colaboratory (Google Colab)** con Python v3.12.

### Fase 1 — ETL y Análisis Exploratorio de Datos (EDA)

#### 1.1 Extracción de Datos
- Extracción automatizada mediante la **API Socrata (SODA)** usando la librería `sodapy`.
- Paginación dinámica con límite de 1.000 registros por petición y *offset* incremental para garantizar la descarga completa del dataset.
- Transformación de los registros JSON a estructura tabular (`DataFrame`) con `pandas`.
- Exportación a archivo CSV con codificación `utf-8-sig`.

#### 1.2 Limpieza y Transformación de Datos
- Los valores faltantes estaban codificados como la etiqueta textual `'NO DISPONIBLE'`, los cuales fueron normalizados a formato `NaN`.
- Se aplicó eliminación de registros con datos nulos, representando menos del **2%** del volumen total, garantizando la integridad estadística del estudio.

#### 1.3 Normalización Semántica de Variables
- La variable `TIPO_ERC` presentaba múltiples etiquetas para una misma modalidad clínica. Se realizó estandarización semántica agrupando variantes textuales bajo categorías unificadas (ej. `HEMODIALISIS`, `DIALISIS PERITONEAL`, `PREDIALISIS`, `TMND`).

#### 1.4 Codificación de Variables
- Variables binarias: codificación `0/1` (SEXO, ARTRITIS, DIABETES, HIPERTENSIÓN, EPOC, ASMA, IC, CÁNCER, ERC, entre otras).
- Variables multiclase (`ciclo_de_vida`, `tipo_erc_limpio`, `erc_trr_limpio`, `régimen`): se aplicó **One-Hot Encoding** con parámetro `drop='first'` para evitar multicolinealidad perfecta.

#### 1.5 Validación de Relevancia Estadística
- Se utilizó la **prueba de independencia Chi-cuadrado (χ²) de Pearson** y el cálculo de **Información Mutua (MI)** para evaluar la asociación de cada atributo con la variable objetivo (`DIABETES`).
- La mayoría de predictores presentaron significancia estadística alta (`p < 0.001`). Variables como `SEXO` (p=0.259) e `INSUFICIENCIA CARDIACA` (p=0.189) se mantuvieron para permitir que los modelos de ensamble capturen interacciones multivariadas no lineales.

---

### Fase 2 — Entrenamiento y Evaluación de Modelos

#### 2.1 Partición de Datos
- División: **70% entrenamiento / 30% prueba** usando `GroupShuffleSplit` para bloquear grupos de perfiles similares y evitar fuga de información (data leakage).

#### 2.2 Tratamiento del Desbalance de Clases — SMOTE
- Se aplicó **SMOTE (Synthetic Minority Over-sampling Technique)** integrado dentro de **Pipelines de Imbalanced-learn**, asegurando que el sobremuestreo se aplique exclusivamente dentro de los pliegues de entrenamiento durante la validación cruzada, manteniendo el conjunto de prueba íntegro con su distribución original.

#### 2.3 Algoritmos Evaluados y Optimización de Hiperparámetros
Se evaluó un espectro multimodal de algoritmos:

| Tipo | Algoritmo |
|------|-----------|
| Modelo Lineal | Regresión Logística (RL) |
| Árbol de Decisión | Decision Tree (DT) |
| Ensamble | Random Forest (RF) |
| Ensamble | Gradient Boosting (GB) |
| Ensamble | XGBoost (XGB) |

Optimización mediante **GridSearchCV** con validación cruzada de 5 pliegues (`k=5`), usando **F1-Score** como métrica principal.

#### 2.4 Análisis de Umbrales de Decisión
Se evaluaron umbrales de clasificación variables (de 0.70 a 0.10) para determinar el **punto óptimo de operación clínica**, priorizando el Recall (sensibilidad) para capturar la mayor cantidad de pacientes en riesgo.

#### 2.5 Interpretabilidad — Importancia de Características (XGBoost)
Se utilizó la métrica de **Ganancia (Gain)** sobre el modelo XGBoost para cuantificar la contribución relativa de cada variable predictora, dotando al modelo de explicabilidad clínica.

---

## 📈 Resultados Principales

### Comparación de Modelos (Umbral = 0.5)

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| XGBoost | — | — | 0.919 | ~0.530 | **0.676** |
| Gradient Boosting | — | — | 0.914 | ~0.529 | 0.671 |
| Random Forest | — | — | 0.930 | ~0.530 | — |
| Árbol de Decisión | **0.735** | **0.690** | 0.354 | — | 0.669 |
| Regresión Logística | 0.530 | — | 0.666 | 0.483 | — |

### Optimización de Umbral — XGBoost
- Ajustando el umbral de decisión a **0.4**, el modelo alcanzó un **Recall de 0.958**, garantizando la identificación de casi la totalidad de individuos en riesgo y minimizando los falsos negativos.

### Variables Más Importantes (XGBoost — Gain)

| Variable | Importancia Relativa |
|----------|----------------------|
| Persona Mayor (ciclo de vida) | 18.47% |
| Cáncer | 9.69% |
| Adultez (ciclo de vida) | 8.02% |
| Asma | 8.02% |
| EPOC | 6.95% |
| Hipertensión | 6.13% |
| Artritis | 5.89% |

---

## 🧪 Tecnologías Utilizadas

- **Lenguaje:** Python 3.12
- **Entorno:** Google Colaboratory (Colab)
- **Librerías principales:** `pandas`, `sodapy`, `scikit-learn`, `imbalanced-learn`, `xgboost`
- **Almacenamiento de datos:** Google Drive

---

## 🏁 Conclusiones

1. **XGBoost** fue el modelo con mayor capacidad discriminativa global (**AUC-ROC = 0.676**) y el más adecuado para detección temprana de diabetes en contextos de salud pública.
2. El ajuste del **umbral de decisión a 0.4** permitió alcanzar un **recall de 0.958**, priorizando la detección sobre la precisión global, lo cual es crítico en tamizaje poblacional donde un falso negativo tiene mayor costo clínico y humano que un falso positivo.
3. El uso de **SMOTE** es esencial al procesar registros administrativos gubernamentales desbalanceados, permitiendo que el modelo identifique patrones de riesgo sin sesgo por la clase mayoritaria.
4. Los predictores más influyentes fueron el **ciclo de vida (edad avanzada)** y comorbilidades como **cáncer, asma, EPOC, hipertensión y artritis**, coherentes con la evidencia epidemiológica disponible.
5. La implementación en **Google Colab** valida la viabilidad tecnológica de desplegar Sistemas de Soporte a la Decisión Clínica (CDSS) sin requerir infraestructuras locales costosas, facilitando la democratización de herramientas de IA en ciudades intermedias.

---

## 📄 Referencia de la Fuente de Datos

Alcaldía de Bucaramanga. (2026). *39. Enfermedades crónicas en el municipio de Bucaramanga*. Datos Abiertos Colombia. Recuperado el 23 de febrero de 2026, de [https://www.datos.gov.co/Salud-y-Protecci-n-Social/39-Enfermedades-cr-nicas-en-el-municipio-de-Bucara/4iz7-suhz/about_data](https://www.datos.gov.co/Salud-y-Protecci-n-Social/39-Enfermedades-cr-nicas-en-el-municipio-de-Bucara/4iz7-suhz/about_data)

---

## ⚖️ Conflictos de Interés

El autor declara que no existe ningún conflicto de interés de tipo financiero, profesional o personal que haya influido de manera inapropiada en el desarrollo de la investigación, en el análisis de los resultados obtenidos o en las interpretaciones propuestas en el presente manuscrito.

---

## 🤝 Contribución de Autoría

**Juan Manuel Betancur:** Responsable de la conceptualización de la investigación, el diseño metodológico, la extracción y el procesamiento de los datos abiertos. Realizó la programación de los modelos de aprendizaje supervisado en Python, la validación cruzada y la optimización de hiperparámetros. Asimismo, se encargó de la redacción del manuscrito original, la interpretación de los resultados clínicos y la revisión final del texto.

---

*Investigación realizada en el marco de la Especialización en Analítica de Datos — Corporación Unificada Nacional de Colombia (CUN).*
