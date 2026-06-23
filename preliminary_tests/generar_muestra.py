import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent 
DATA_DIR = BASE_DIR / "train_corpora"

archivo_train = DATA_DIR / "train_public.csv"
print(f"Cargando dataset original desde: {archivo_train}")

df_train = pd.read_csv(archivo_train)
print(f"Dataset original cargado: {len(df_train)} filas.")

TAMANO_MUESTRA = 1000
df_sample = df_train.sample(n=TAMANO_MUESTRA, random_state=42).reset_index(drop=True)

archivo_salida = "train_sample_1000.csv"
df_sample.to_csv(archivo_salida, index=False)

print("\n" + "="*40)
print(f"Archivo guardado como: {archivo_salida}")
print(f"Total de noticias en la muestra: {len(df_sample)}")
print("="*40)