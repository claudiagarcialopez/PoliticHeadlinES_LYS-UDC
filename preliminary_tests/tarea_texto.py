import os
import pandas as pd
import numpy as np
import re
import time
from groq import Groq
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "train_corpora" 

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)
MODELO = "llama-3.3-70b-versatile" 

def predict_headline_ranking(body, titles):
    """
    Solicita al LLM que ordene los 10 titulares por relevancia.
    """
    titulares_formateados = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])

    prompt = f"""Eres un experto en política española. Tu tarea es ordenar 10 titulares del más probable al menos probable basándote en el cuerpo de la noticia.
    
    INSTRUCCIÓN ESTRICTA: Responde EXCLUSIVAMENTE con una lista de números separados por comas y encerrados entre corchetes. 
    No añadas saludos, ni introducciones, ni etiquetas de pensamiento. 
    
    Ejemplo de respuesta válida: [3, 5, 1, 10, 2, 4, 8, 7, 6, 9]
    """    

    user_prompt = f"CUERPO:\n{body[:2000]}\n\nTITULARES:\n{titulares_formateados}"
    try:
        completion = client.chat.completions.create(
            model=MODELO,
            messages=[{"role": "user", "content": prompt}, {"role": "user", "content": user_prompt}],
            temperature=0, 
            max_tokens=1500
        )
        response = completion.choices[0].message.content.strip()
        print(f"Respuesta cruda del modelo: {response}")
        
        match = re.search(r'\[(.*?)\]', response)
        if match:
            nums = [n.strip() for n in match.group(1).split(',')]
            return nums
        return None
    except Exception as e:
        if "429" in str(e):
            time.sleep(10)
            return predict_headline_ranking(body, titles)
        
        print(f"\nError de la API: {e}")
        return None

def calculate_mrr(target, ranking):
    """Calcula el Mean Reciprocal Rank"""
    try:
        rank = ranking.index(str(target)) + 1
        return 1.0 / rank
    except (ValueError, AttributeError):
        return 0.0

def calculate_ndcg(target, ranking):
    """Calcula el NDCG simplificado para un solo item relevante"""
    try:
        rank = ranking.index(str(target)) + 1
        return 1.0 / np.log2(rank + 1)
    except (ValueError, AttributeError):
        return 0.0

if __name__ == "__main__":
    archivo_muestra = os.path.join(DATA_DIR, "train_sample_1000.csv")
        
    if not os.path.exists(archivo_muestra):
        raise FileNotFoundError(f"No se encuentra '{archivo_muestra}'.")

    df_train = pd.read_csv(archivo_muestra)
    sample_df = df_train.copy()

    rankings = []
    print(f"Iniciando procesado de {len(sample_df)} noticias.")
    print(f"Iniciando ranking con {MODELO}...")

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        titles = [row[f'title_{i}'] for i in range(1, 11)]
        body = row['article_body'] 
        
        rank = predict_headline_ranking(body, titles)
        rankings.append(rank)
        time.sleep(1) # Respeto de cuota

    sample_df['pred_ranking'] = rankings
    sample_df['target_num'] = sample_df['y_true'].str.extract(r'(\d+)')

    # Cálculo de Métricas
    sample_df['mrr'] = sample_df.apply(lambda x: calculate_mrr(x['target_num'], x['pred_ranking']), axis=1)
    sample_df['ndcg'] = sample_df.apply(lambda x: calculate_ndcg(x['target_num'], x['pred_ranking']), axis=1)
    sample_df['hit_at_3'] = sample_df.apply(lambda x: 1.0 if str(x['target_num']) in (x['pred_ranking'][:3] if x['pred_ranking'] else []) else 0.0, axis=1)

    print("\n--- Métricas ---")
    print(f"Accuracy (Top-1): { (sample_df['mrr'] == 1.0).mean() * 100:.2f}%")
    print(f"Mean MRR: {sample_df['mrr'].mean():.4f}")
    print(f"Mean NDCG: {sample_df['ndcg'].mean():.4f}")
    print(f"Hit Rate @3: {sample_df['hit_at_3'].mean() * 100:.2f}%")

    output_file = "results/resultados_texto.csv"
    sample_df.to_csv(output_file, index=False)
    print(f"\nResultados guardados con éxito en '{output_file}'")