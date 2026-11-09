"""Funciones compartidas por el entrenamiento, las predicciones nuevas y la automatización de SaludConex."""
import numpy as np
import pandas as pd
from scipy.special import logit

ETIQUETAS = {
    'edad': 'Edad', 'dias_anticipacion': 'Días entre reserva y cita',
    'asistencias_anteriores': 'Asistencias anteriores', 'inasistencias_anteriores': 'Inasistencias anteriores',
    'cancelaciones_anteriores': 'Cancelaciones anteriores', 'reservas_previas': 'Reservas previas',
    'tasa_inasistencia_hist': 'Tasa histórica de inasistencia', 'hora_atencion': 'Hora de la cita',
    'dia_semana': 'Día de la cita', 'mes': 'Mes de la cita', 'centro_medico': 'Centro',
    'especialidad': 'Especialidad', 'modalidad_atencion': 'Modalidad',
    'confirmacion_recordatorio': 'Recordatorio'}
DIAS = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']


def crear_variables(d):
    """Variables derivadas. Se usa igual para entrenamiento, nuevas predicciones y la automatización."""
    d = d.copy()
    d['hora_atencion'] = d['horario_atencion'].astype(str).str[:2].astype(int)
    d['dia_semana'] = d['fecha_atencion'].dt.dayofweek
    d['mes'] = d['fecha_atencion'].dt.month
    d['reservas_previas'] = d['asistencias_anteriores'] + d['inasistencias_anteriores'] + d['cancelaciones_anteriores']
    atendidas = d['asistencias_anteriores'] + d['inasistencias_anteriores']
    d['tasa_inasistencia_hist'] = np.where(atendidas > 0, d['inasistencias_anteriores'] / atendidas.where(atendidas > 0, 1), 0)
    return d


def preparar_reservas(d, fecha_proceso):
    """Construye la foto conocida al corte; el historial de entrada debe corresponder a ese corte."""
    d = d.drop_duplicates('id_reserva').copy()
    corte = pd.Timestamp(fecha_proceso)
    for c in ['fecha_atencion', 'fecha_solicitud', 'fecha_hora_solicitud', 'fecha_cancelacion',
              'fecha_confirmacion', 'fecha_prediccion']:
        d[c] = pd.to_datetime(d[c])
    activa = d.fecha_cancelacion.isna() | d.fecha_cancelacion.gt(corte)
    d = d[d.fecha_hora_solicitud.le(corte) & activa].copy()
    # No aceptar antecedentes calculados para otro instante.
    if not d.empty and not d.fecha_prediccion.eq(corte).all():
        raise ValueError('El historial debe calcularse al corte exacto de esta ejecución')
    conf = d.fecha_confirmacion.notna() & d.fecha_confirmacion.le(corte)
    falta = d.confirmacion_recordatorio.isna()
    d['confirmacion_recordatorio'] = pd.Series(np.where(conf, 'Confirmó', 'Sin confirmación al corte'),
                                               index=d.index, dtype=object)
    d.loc[falta, 'confirmacion_recordatorio'] = np.nan
    return crear_variables(d)


def variables_modelo(paquete):
    return paquete['variables_numericas'] + paquete['variables_categoricas']


def probabilidad(paquete, X):
    """Probabilidad calibrada de inasistencia con el pipeline y el calibrador del paquete."""
    p = paquete['pipeline'].predict_proba(X)[:, 1]
    if paquete.get('calibrador') is not None:
        p = paquete['calibrador'].predict_proba(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]
    return p


def nivel_riesgo(paquete, p):
    p = np.asarray(p)
    return np.where(p >= paquete['umbral_alto'], 'Alto', np.where(p >= paquete['umbral_medio'], 'Medio', 'Bajo'))


def _denso(z):
    return z.toarray() if hasattr(z, 'toarray') else np.asarray(z)


def _grupos_columnas(paquete):
    """Índices de las columnas transformadas que pertenecen a cada variable original."""
    nombres = list(paquete['pipeline'].named_steps['prep'].get_feature_names_out())
    grupos = {}
    for v in paquete['variables_numericas']:
        grupos[v] = [nombres.index(f'num__{v}')]
    for v in paquete['variables_categoricas']:
        if f'cat__{v}' in nombres:                       # codificación ordinal: una columna
            grupos[v] = [nombres.index(f'cat__{v}')]
        else:                                            # one-hot: varias columnas
            grupos[v] = [i for i, n in enumerate(nombres) if n.startswith(f'cat__{v}_')]
    return grupos


def _formato(var, valor):
    if pd.isna(valor):
        return 'sin dato'
    if var == 'tasa_inasistencia_hist':
        return f'{valor:.0%}'
    if var == 'dia_semana':
        return DIAS[int(valor)]
    if var == 'hora_atencion':
        return f'{int(valor):02d}:00'
    if isinstance(valor, (int, float, np.integer, np.floating)):
        return str(int(valor)) if float(valor).is_integer() else f'{valor:.1f}'
    return str(valor)


def crear_explicador(paquete):
    """Explicador SHAP construido solo con el contenido del paquete (mismo fondo en todo el proyecto)."""
    import shap
    estimador = paquete['pipeline'].named_steps['modelo']
    fondo = paquete['fondo_shap']
    if paquete['algoritmo'] == 'Regresión logística':
        return shap.LinearExplainer(estimador, fondo)
    return shap.PermutationExplainer(estimador.decision_function,
                                     shap.maskers.Independent(fondo, max_samples=len(fondo)),
                                     seed=paquete['semilla'])


def explicar(paquete, d, explicador=None, top=3):
    """Devuelve (textos legibles, contribuciones por variable original, valor base).

    Las contribuciones están en log-odds del score bruto (antes de calibrar): indican
    qué factores empujan el riesgo hacia arriba o hacia abajo, no efectos causales.
    """
    explicador = explicador or crear_explicador(paquete)
    preprocesador = paquete['pipeline'].named_steps['prep']
    estimador = paquete['pipeline'].named_steps['modelo']
    z = _denso(preprocesador.transform(d[variables_modelo(paquete)]))
    if paquete['algoritmo'] == 'Regresión logística':
        valores = np.asarray(explicador.shap_values(z))
        base = float(np.ravel(explicador.expected_value)[0])
    else:
        e = explicador(z, max_evals=5 * (2 * z.shape[1] + 1), silent=True)
        valores, base = np.asarray(e.values), float(np.ravel(e.base_values)[0])
    assert np.allclose(base + valores.sum(axis=1), estimador.decision_function(z), atol=1e-4), 'Revisar suma de SHAP'
    grupos = _grupos_columnas(paquete)
    agregados = pd.DataFrame({v: valores[:, idx].sum(axis=1) for v, idx in grupos.items()}, index=d.index)
    textos = []
    for k, (_, fila) in enumerate(agregados.iterrows()):
        partes = []
        for v in fila.abs().sort_values(ascending=False).index[:top]:
            flecha = '↑ riesgo' if fila[v] > 0 else '↓ riesgo'
            partes.append(f'{ETIQUETAS.get(v, v)}: {_formato(v, d[v].iloc[k])} ({flecha})')
        textos.append('; '.join(partes))
    return textos, agregados, base
