"""Proceso diario SaludConex (20:00): lista priorizada de contactos preventivos para la sucursal piloto.

No contacta, no cancela y no modifica reservas. Solo necesita el paquete del modelo, su configuración
y el módulo compartido saludconex_variables.py.

Uso:
    python automatizacion.py --datos saludconex_reservas_2023_2025.csv --corte "2025-11-30 20:00"
"""
import argparse
import json
import os
from datetime import datetime

import joblib
import pandas as pd

from saludconex_variables import preparar_reservas, probabilidad, nivel_riesgo, explicar, variables_modelo

FECHAS = ['fecha_solicitud', 'fecha_atencion', 'fecha_hora_atencion', 'fecha_hora_solicitud',
          'fecha_prediccion', 'fecha_cancelacion', 'fecha_confirmacion']
COLUMNAS_SALIDA = ['id_reserva', 'id_paciente', 'centro_medico', 'fecha_atencion', 'horario_atencion',
                   'prob_inasistencia', 'nivel_riesgo', 'accion_sugerida', 'factores_principales', 'version_modelo']


def cargar_modelo(version, carpeta_modelo='modelo'):
    with open(f'{carpeta_modelo}/config_{version}.json', encoding='utf-8') as f:
        cfg = json.load(f)
    paquete = joblib.load(f'{carpeta_modelo}/modelo_inasistencia_{version}.joblib')
    return cfg, paquete


def generar_lista_acciones(reservas, fecha_proceso, version='v2.1', carpeta='salidas', carpeta_modelo='modelo'):
    """Genera la lista diaria. No accede al resultado final, no contacta ni cancela."""
    cfg, paquete = cargar_modelo(version, carpeta_modelo)
    corte = pd.Timestamp(fecha_proceso)
    if corte.hour != cfg['hora_proceso'] or corte.minute or corte.second:
        raise ValueError(f"El proceso diario requiere corte a las {cfg['hora_proceso']}:00")
    fecha_obj = corte.normalize() + pd.Timedelta(days=1)

    dia = reservas[pd.to_datetime(reservas.fecha_atencion).eq(fecha_obj) &
                   reservas.centro_medico.eq(cfg['sucursal_piloto'])].copy()
    # Eliminar la etiqueta si viniera en datos retrospectivos; nunca se consulta para decidir.
    dia = preparar_reservas(dia.drop(columns=['resultado_reserva'], errors='ignore'), corte)
    evaluables = len(dia)
    if not dia.empty:
        dia['prob_inasistencia'] = probabilidad(paquete, dia[variables_modelo(paquete)])
        dia['nivel_riesgo'] = nivel_riesgo(paquete, dia['prob_inasistencia'])
        dia = (dia[dia.nivel_riesgo.isin(['Alto', 'Medio'])]
               .sort_values(['prob_inasistencia', 'id_reserva'], ascending=[False, True])
               .head(cfg['capacidad_diaria']).copy())
    if dia.empty:
        salida = pd.DataFrame(columns=COLUMNAS_SALIDA)
    else:
        dia['accion_sugerida'] = dia['nivel_riesgo'].map(paquete['acciones'])
        dia['factores_principales'], _, _ = explicar(paquete, dia)
        dia['version_modelo'] = version
        salida = dia[COLUMNAS_SALIDA].reset_index(drop=True)

    os.makedirs(carpeta, exist_ok=True)
    archivo = f"{carpeta}/acciones_preventivas_{cfg['sucursal_piloto']}_{fecha_obj.date()}.csv"
    salida.to_csv(archivo, index=False)
    conteo = salida['nivel_riesgo'].value_counts().to_dict()
    log = {'ejecucion_real': datetime.now().isoformat(timespec='seconds'), 'corte_simulado': str(corte),
           'fecha_atencion': str(fecha_obj.date()), 'version_modelo': version, 'sucursal': cfg['sucursal_piloto'],
           'capacidad_diaria': cfg['capacidad_diaria'], 'reservas_evaluables': evaluables,
           'contactos': len(salida), 'alto': int(conteo.get('Alto', 0)), 'medio': int(conteo.get('Medio', 0)),
           'archivo': archivo}
    ruta_log = f'{carpeta}/log_ejecuciones.csv'
    logs = pd.read_csv(ruta_log) if os.path.exists(ruta_log) else pd.DataFrame()
    logs = pd.concat([logs, pd.DataFrame([log])], ignore_index=True)
    logs = logs.drop_duplicates(['corte_simulado', 'version_modelo', 'sucursal'], keep='last')
    logs.to_csv(ruta_log, index=False)
    return salida, log


def main():
    parser = argparse.ArgumentParser(description='Lista diaria de contactos preventivos SaludConex')
    parser.add_argument('--datos', required=True, help='CSV de reservas con historial calculado al corte')
    parser.add_argument('--corte', required=True, help='Fecha y hora del corte, p. ej. "2025-11-30 20:00"')
    parser.add_argument('--version', default='v2.1')
    parser.add_argument('--salida', default='salidas')
    args = parser.parse_args()
    reservas = pd.read_csv(args.datos, parse_dates=FECHAS)
    salida, log = generar_lista_acciones(reservas, args.corte, args.version, args.salida)
    print(json.dumps(log, ensure_ascii=False))
    print(salida[['id_reserva', 'horario_atencion', 'prob_inasistencia', 'nivel_riesgo']].head(5).to_string(index=False))


if __name__ == '__main__':
    main()
