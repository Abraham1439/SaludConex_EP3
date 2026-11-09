# SaludConex · Predicción de inasistencia a consultas médicas

Prototipo académico del proyecto semestral de **SCY1102 Gestión de Proyectos de Datos** (Duoc UC, 2026).
Equipo: Abraham Abanto, Bastián Soto y Andrés Silva.

SaludConex pierde el 17% de sus horas médicas por inasistencia. Este prototipo estima la probabilidad de que cada reserva termine en inasistencia y genera, cada día a las 20:00, una lista priorizada de hasta 30 pacientes para contacto preventivo en la sede piloto. **El sistema nunca cancela ni modifica una hora**: solo ordena a quién dedicar el esfuerzo humano, y la decisión queda en Admisión.

> Todos los datos son **sintéticos** y se generan con el notebook 01. No hay datos reales de pacientes en este repositorio.

## Resultados principales (test 2025, datos sintéticos)

| Indicador | Valor |
| --- | --- |
| Modelo seleccionado | Regresión logística calibrada |
| AUC-ROC | 0,742 (referencia Dummy: 0,500) |
| Calibración | 18,59% predicho vs 18,45% real |
| Inasistencias anticipadas con 30 llamadas/día (sede piloto) | 56,9% |
| Brecha de falsos negativos por edad | 0,151 con umbral alto único → 0,025 con umbrales por edad (sección 6.1) |

## Estructura

```
notebooks/
  01_SaludConex_Generacion_Datos_Sinteticos.ipynb   # genera saludconex_reservas_2023_2025.csv (1.536.727 reservas)
  02_SaludConex_Actividad_3_1_2.ipynb               # preparación, modelo, evaluación, equidad, predicciones, automatización y CSV para Power BI
src/
  saludconex_variables.py   # módulo compartido: variables, corte 20:00, probabilidad, nivel de riesgo, explicaciones SHAP
  automatizacion.py         # proceso diario ejecutable por línea de comandos
requirements.txt
.gitignore
```

Los archivos de `src/` son una copia de los que el notebook 02 escribe con `%%writefile`, para poder revisarlos sin abrir el notebook.

## Cómo ejecutarlo

### En Google Colab

1. Ejecuta el notebook 01 y descarga `saludconex_reservas_2023_2025.csv` (~340 MB).
2. Abre el notebook 02, sube el CSV y **espera a que termine la subida** (o léelo desde Google Drive).
3. Ejecuta el notebook 02 completo.

### En tu computador (requiere Python 3.10 a 3.12)

```bash
# 1. Crear el entorno virtual (desde la carpeta raíz del repositorio)
python -m venv .venv

# 2. Activarlo
#    Windows:      .venv\Scripts\activate
#    macOS/Linux:  source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Abrir los notebooks
cd notebooks
jupyter notebook
```

Ejecuta primero el notebook 01 y luego el 02 completo, ambos en `notebooks/`, para que el CSV quede en la misma carpeta que el notebook 02. Al terminar, el notebook 02 genera en esa carpeta:

- `modelo/modelo_inasistencia_v2.1.joblib` y `modelo/config_v2.1.json`
- `salidas/acciones_preventivas_Providencia_<fecha>.csv` y `salidas/log_ejecuciones.csv`
- `saludconex_dashboard_powerbi.csv` (Paso 9): 510.047 reservas de 2025, incluidas las canceladas, con la predicción del modelo para las 406.821 evaluables. Separador `;` y coma decimal, listo para Power BI en español.

Los CSV, el modelo y las salidas no se suben al repositorio (ver `.gitignore`): se recrean al ejecutar los notebooks.

El proceso diario también se puede correr sin abrir el notebook, desde `notebooks/` y con el entorno activado:

```bash
python automatizacion.py --datos saludconex_reservas_2023_2025.csv --corte "2025-11-30 20:00"
```

## Decisiones técnicas

- **Validación temporal:** entrenamiento hasta jun-2024, selección jul–sep 2024, calibración oct–dic 2024 y test 2025. El test no se usa para elegir el modelo ni los umbrales.
- **Sin información del futuro:** solo reservas vigentes al corte de las 20:00 del día anterior y confirmaciones registradas hasta ese instante.
- **Variables:** agendamiento, historial y confirmación. Sin datos clínicos, previsión ni comuna. Edad y centro médico se excluyeron por una prueba de necesidad (aportaban menos de 0,01 de AUC).
- **Selección:** regresión logística y Gradient Boosting empataron (diferencia 0,0006); se eligió la regresión logística por simplicidad y explicabilidad.
- **Equidad:** auditoría de falsos positivos y negativos por edad, comuna y sede (RNF10). La sección 6.1 evalúa umbrales por grupo de edad como acción correctiva.

## Gestión del proyecto

- Tablero de control (Trello): https://trello.com/b/38zD2Typ/caso-semestral-saludconex
- Notebook 01 en Colab: https://colab.research.google.com/drive/1sfmXiYUNonMR4znNIT896YDlbaOrAoua?usp=sharing
- Notebook 02 en Colab: https://colab.research.google.com/drive/1NnGUGzqpo-L8c7V9FuRvIRP9OSLb9Z4L?usp=sharing

## Limitaciones

Los resultados demuestran el proceso técnico, no el impacto real en SaludConex. El efecto sobre la inasistencia debe medirse en un piloto con datos reales y una sede de control comparable. La conexión directa a SQL, la publicación del dashboard, el cifrado y los accesos por rol dependen de la infraestructura institucional y no están implementados aquí.