import os
import pandas as pd
import numpy as np
import re
import time
import base64
from groq import Groq
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "train_corpora"

IMAGE_DIR = DATA_DIR / "images" 

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("¡Error! No se ha encontrado la GROQ_API_KEY en las variables de entorno.")

client = Groq(api_key=api_key)

# Modelo con capacidades de visión 
MODELO_VISION = "meta-llama/llama-4-scout-17b-16e-instruct" 

def encode_image(image_path):
    """Lee una imagen local y la convierte a base64. Retorna None si falla o no existe."""
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"\nError al leer la imagen {image_path}: {e}")
        return None

def predict_headline_ranking_multimodal(body, titles, image_path):
    """
    Solicita al LLM (Visión) que ordene los 10 titulares usando texto E imagen.
    """
    titulares_formateados = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])

    # Instrucciones del sistema
    prompt_instrucciones = """Eres un experto en política española y edición periodística. 
    Tu tarea es ordenar 10 titulares del más probable al menos probable basándote en el CUERPO de la noticia y en la IMAGEN adjunta (si la hay).
    
    INSTRUCCIÓN ESTRICTA: Responde EXCLUSIVAMENTE con una lista de números separados por comas y encerrados entre corchetes. 
    Ejemplo de respuesta válida: [3, 5, 1, 10, 2, 4, 8, 7, 6, 9]"""    

    # Recorte del cuerpo a 1500 caracteres para no saturar los tokens con la imagen
    user_text = f"CUERPO:\n{body[:1500]}\n\nTITULARES:\n{titulares_formateados}"
    
    # Procesar la imagen
    base64_image = encode_image(image_path)
    
    # Construir el contenido del usuario mezclando texto e imagen
    user_content = [{"type": "text", "text": user_text}]
    if base64_image:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    try:
        completion = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[
                {"role": "system", "content": prompt_instrucciones},
                {"role": "user", "content": user_content}
            ],
            temperature=0, 
            max_tokens=50 
        )
        response = completion.choices[0].message.content.strip()
        print(f"Respuesta cruda del modelo: {response}")
        
        match = re.search(r'\[(.*?)\]', response)
        if match:
            return [n.strip() for n in match.group(1).split(',')]
        
        # Fallback por si el LLM no pone los corchetes
        nums_fallback = re.findall(r'\d+', response)
        if len(nums_fallback) >= 1:
            vistos = set()
            resultado = []
            for n in nums_fallback:
                if n in [str(i) for i in range(1, 11)] and n not in vistos:
                    resultado.append(n)
                    vistos.add(n)
            return resultado if len(resultado) > 0 else None
            
        return None
        
    except Exception as e:
        if "429" in str(e):
            time.sleep(15) # Pausa más larga por los límites de visión
            return predict_headline_ranking_multimodal(body, titles, image_path)
        print(f"\nError de la API: {e}")
        return None

def calculate_mrr(target, ranking):
    """Calcula el Mean Reciprocal Rank"""
    if not ranking: return 0.0
    try:
        rank = ranking.index(str(target)) + 1
        return 1.0 / rank
    except (ValueError, AttributeError):
        return 0.0

def calculate_ndcg(target, ranking):
    """Calcula el NDCG simplificado para un solo item relevante"""
    if not ranking: return 0.0
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
    
    sample_df = df_train.head(500).copy()

    rankings = []
    print(f"Iniciando procesado multimodal de {len(sample_df)} noticias.")
    print(f"Modelo: {MODELO_VISION}")

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        titles = [row[f'title_{i}'] for i in range(1, 11)]
        body = row['article_body']
        
        nombre_imagen = str(row.get('image_hash', f"{idx}.jpg"))
        ruta_imagen = os.path.join(IMAGE_DIR, nombre_imagen)
        
        rank = predict_headline_ranking_multimodal(body, titles, ruta_imagen)
        rankings.append(rank)
        
        time.sleep(2) 

    # Guardar resultados
    sample_df['pred_ranking'] = rankings
    sample_df['target_num'] = sample_df['y_true'].str.extract(r'(\d+)')

    # Cálculo de Métricas
    sample_df['mrr'] = sample_df.apply(lambda x: calculate_mrr(x['target_num'], x['pred_ranking']), axis=1)
    sample_df['ndcg'] = sample_df.apply(lambda x: calculate_ndcg(x['target_num'], x['pred_ranking']), axis=1)
    sample_df['hit_at_3'] = sample_df.apply(lambda x: 1.0 if str(x['target_num']) in (x['pred_ranking'][:3] if x['pred_ranking'] else []) else 0.0, axis=1)

    print("\n--- Métricas ---")
    print(f" Accuracy (Top-1): { (sample_df['mrr'] == 1.0).mean() * 100:.2f}%")
    print(f" Mean MRR:         {sample_df['mrr'].mean():.4f}")
    print(f" Mean NDCG:        {sample_df['ndcg'].mean():.4f}")
    print(f" Hit Rate @3:      {sample_df['hit_at_3'].mean() * 100:.2f}%")

    output_file = "results/resultados_multimodal.csv"
    sample_df.to_csv(output_file, index=False)
    print(f"\nResultados guardados con éxito en '{output_file}'")