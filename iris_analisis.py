# Iris Dataset — Análisis de Datos

import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/Cruz/Downloads/archive/Iris.csv")

print("="*60)
print("1. ESTRUCTURA DEL DATASET")
print("="*60)
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
print(f"\nColumnas: {df.columns.tolist()}")

print("\n" + "="*60)
print("2. ESTADÍSTICAS DESCRIPTIVAS")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("3. DISTRIBUCIÓN POR ESPECIE")
print("="*60)
print(df["Species"].value_counts())

print("\n" + "="*60)
print("4. MÉTRICAS POR ESPECIE")
print("="*60)
metricas = df.groupby("Species").agg({
    "SepalLengthCm": ["mean", "std"],
    "SepalWidthCm": ["mean", "std"],
    "PetalLengthCm": ["mean", "std"],
    "PetalWidthCm": ["mean", "std"]
}).round(2)
print(metricas)

print("\n" + "="*60)
print("5. ANÁLISIS DE CORRELACIÓN")
print("="*60)
numeric_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
corr = df[numeric_cols].corr().round(2)
print(corr)

print("\n" + "="*60)
print("6. OBSERVACIONES CLAVE")
print("="*60)
print("• Setosa: pétalos pequeños (longitud media ~1.46cm)")
print("• Versicolor: pétalos medianos (longitud media ~4.26cm)")
print("• Virginica: pétalos grandes (longitud media ~5.55cm)")
print("• Los pétalos son el mejor discriminante entre especies")