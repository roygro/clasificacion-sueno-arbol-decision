# ============================================================
# GENERADOR DE DATASET PROPIO - HÁBITOS DE SUEÑO
# Riesgo de Insomnio: Sí / No
# ============================================================
# Dataset de creación propia (Actividad 2), con apego a una
# actividad cotidiana: hábitos de sueño y descanso.
# Se genera con reglas lógicas + ruido aleatorio, simulando
# relaciones reales entre hábitos diarios y riesgo de insomnio.
# ============================================================

import pandas as pd
import numpy as np

np.random.seed(7)
N = 1800  # número de registros a generar

# ------------------------------------------------------------
# Categorías posibles para cada variable
# ------------------------------------------------------------
consumo_cafeina_opts = ['ninguna', 'baja', 'media', 'alta']
pantalla_antes_dormir_opts = ['ninguno', 'poco', 'moderado', 'mucho']
ejercicio_opts = ['nunca', 'ocasional', 'frecuente']
estres_opts = ['bajo', 'medio', 'alto']
siestas_opts = ['ninguna', 'una', 'dos_o_mas']
ruido_opts = ['bajo', 'medio', 'alto']
luz_opts = ['oscura', 'tenue', 'iluminada']
temperatura_opts = ['fria', 'templada', 'calurosa']
horario_trabajo_opts = ['diurno', 'nocturno', 'rotativo']
si_no = ['sí', 'no']

registros = []

for i in range(N):
    horas_sueno = round(np.clip(np.random.normal(6.5, 1.3), 3, 10), 1)
    cafeina = np.random.choice(consumo_cafeina_opts, p=[0.25, 0.35, 0.25, 0.15])
    pantalla = np.random.choice(pantalla_antes_dormir_opts, p=[0.2, 0.3, 0.3, 0.2])
    ejercicio = np.random.choice(ejercicio_opts, p=[0.3, 0.4, 0.3])
    estres = np.random.choice(estres_opts, p=[0.3, 0.4, 0.3])
    ruido = np.random.choice(ruido_opts, p=[0.4, 0.4, 0.2])
    luz = np.random.choice(luz_opts, p=[0.4, 0.35, 0.25])
    temperatura = np.random.choice(temperatura_opts, p=[0.25, 0.5, 0.25])
    horario_trabajo = np.random.choice(horario_trabajo_opts, p=[0.6, 0.15, 0.25])
    comida_pesada = np.random.choice(si_no, p=[0.3, 0.7])
    alcohol = np.random.choice(si_no, p=[0.2, 0.8])
    nicotina = np.random.choice(si_no, p=[0.15, 0.85])
    dispositivo_cama = np.random.choice(si_no, p=[0.55, 0.45])
    actividad_relajante = np.random.choice(si_no, p=[0.35, 0.65])
    siestas = np.random.choice(siestas_opts, p=[0.5, 0.35, 0.15])
    despertares = np.random.poisson(1.2)
    dias_horario_irregular = np.random.randint(0, 8)

    # --- Regla de generación del riesgo (probabilística) ---
    riesgo = 0
    if horas_sueno < 6:
        riesgo += 3
    elif horas_sueno < 7:
        riesgo += 1

    if cafeina == 'alta':
        riesgo += 2
    elif cafeina == 'media':
        riesgo += 1

    if pantalla == 'mucho':
        riesgo += 2
    elif pantalla == 'moderado':
        riesgo += 1

    if estres == 'alto':
        riesgo += 2
    elif estres == 'medio':
        riesgo += 1

    if ruido == 'alto':
        riesgo += 1
    if luz == 'iluminada':
        riesgo += 1
    if horario_trabajo in ('nocturno', 'rotativo'):
        riesgo += 2
    if comida_pesada == 'sí':
        riesgo += 1
    if alcohol == 'sí':
        riesgo += 1
    if nicotina == 'sí':
        riesgo += 1
    if dispositivo_cama == 'sí':
        riesgo += 1
    if despertares >= 3:
        riesgo += 2
    if dias_horario_irregular >= 4:
        riesgo += 1

    if ejercicio == 'frecuente':
        riesgo -= 2
    elif ejercicio == 'ocasional':
        riesgo -= 1
    if actividad_relajante == 'sí':
        riesgo -= 1
    if temperatura == 'templada':
        riesgo -= 1

    prob_insomnio = 1 / (1 + np.exp(-(riesgo - 5) * 0.6))  # función logística
    prob_insomnio = np.clip(prob_insomnio, 0.03, 0.95)
    riesgo_insomnio = 'sí' if np.random.rand() < prob_insomnio else 'no'

    registros.append({
        'riesgo_insomnio': riesgo_insomnio,
        'horas_sueno_promedio': horas_sueno,
        'consumo_cafeina': cafeina,
        'uso_pantalla_antes_dormir': pantalla,
        'ejercicio_frecuencia': ejercicio,
        'nivel_estres': estres,
        'siestas_dia': siestas,
        'ruido_ambiente': ruido,
        'luz_habitacion': luz,
        'temperatura_habitacion': temperatura,
        'horario_trabajo': horario_trabajo,
        'comida_pesada_antes_dormir': comida_pesada,
        'consumo_alcohol': alcohol,
        'consumo_nicotina': nicotina,
        'uso_dispositivo_en_cama': dispositivo_cama,
        'actividad_relajante_antes_dormir': actividad_relajante,
        'numero_despertares_noche': despertares,
        'dias_horario_irregular_semana': dias_horario_irregular,
    })

df = pd.DataFrame(registros)
df.to_csv('sueno_dataset_propio.csv', index=False, encoding='utf-8-sig')

print("Dataset generado exitosamente")
print(f"Registros: {df.shape[0]}")
print(f"Variables: {df.shape[1]} (17 predictoras + 1 variable objetivo 'riesgo_insomnio')")
print(f"\nDistribución de clases:")
print(df['riesgo_insomnio'].value_counts())
print(f"\nPrimeras filas:")
print(df.head())
print(f"\nArchivo guardado como: sueno_dataset_propio.csv")
