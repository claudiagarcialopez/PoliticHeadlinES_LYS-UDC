import os
import argparse
import importlib.util
import pandas as pd
import numpy as np
import re
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/test_corpora"


def load_env_file(env_path):
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(BASE_DIR / ".env")

SYSTEM_PROMPT = """Eres un experto en politica española. Tu tarea es ordenar 10 titulares del mas probable al menos probable basandote en el cuerpo de la noticia. Se te va a enviar una NOTICIA y 10 TITULARES POSIBLES. Solo uno de los titulares es el correcto (el que realmente corresponde a la noticia), pero algunos de los otros titulares pueden ser plausibles o parcialmente correctos. Tu objetivo es identificar el titular correcto y ordenarlos por relevancia.

INSTRUCCIÓN ESTRICTA: Responde EXCLUSIVAMENTE con una lista de numeros separados por comas y encerrados entre corchetes, sino la ejecución fallará. No añadas saludos, ni introducciones, ni etiquetas de pensamiento. No repitas los titulares del ranking, solo sus numeros del 1 al 10 segun su posicion. Si el modelo no esta seguro, debes ordenar los que consideres mas probables al principio y luego completar con los numeros restantes en cualquier orden.

Ejemplo de respuesta valida: [3, 5, 1, 10, 2, 4, 8, 7, 6, 9]

RESPUESTA:
"""

# Modelo local por defecto (debe existir en cache/local).
#DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"


class LocalHFLLM:
    def __init__(self, tokenizer, model, max_new_tokens=2048):
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens

    def _build_inputs(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Si el tokenizer soporta plantilla de chat, se usa para mantener formato conversacional.
        if hasattr(self.tokenizer, "apply_chat_template"):
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = f"Sistema:\n{system_prompt}\n\nUsuario:\n{user_prompt}\n\nAsistente:\n"

        return self.tokenizer(rendered, return_tensors="pt")

    def generate(self, system_prompt, user_prompt):
        inputs = self._build_inputs(system_prompt, user_prompt)
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                #temperature=0.0,
                pad_token_id=pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        generated_tokens = output[0][prompt_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def get_quantization_config(quantization):
    if quantization == "none":
        return None

    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)

    raise ValueError(f"Cuantizacion no soportada: {quantization}")


def safe_model_folder_name(model_name_or_path):
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_name_or_path.strip("/"))


def has_accelerate():
    return importlib.util.find_spec("accelerate") is not None


def resolve_model_path(model_name_or_path, download_from_hub=True, cache_dir=None, revision=None, token=None):
    candidate_path = Path(model_name_or_path).expanduser()
    if candidate_path.exists():
        return str(candidate_path.resolve())

    if not download_from_hub:
        return model_name_or_path

    return model_name_or_path


def load_local_hf_model(
    model_name_or_path,
    quantization="none",
    trust_remote_code=False,
    max_new_tokens=2048,
    download_from_hub=True,
    cache_dir=None,
    revision=None,
    token=None,
):
    quant_config = get_quantization_config(quantization)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    use_device_map = quantization != "none"
    if use_device_map and not has_accelerate():
        raise RuntimeError(
            "La carga con cuantizacion requiere 'accelerate'. Instala con: pip install accelerate"
        )
    resolved_model_path = resolve_model_path(
        model_name_or_path,
        download_from_hub=download_from_hub,
        cache_dir=cache_dir,
        revision=revision,
        token=token,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        cache_dir=cache_dir,
        local_files_only=not download_from_hub,
        revision=revision,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "local_files_only": not download_from_hub,
        "dtype": dtype,
        "cache_dir": cache_dir,
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
    }
    if use_device_map:
        model_kwargs["device_map"] = "auto"
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(resolved_model_path, **model_kwargs)

    if not use_device_map:
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(model_device)

    # Evita warnings de flags de muestreo cuando usamos decodificacion greedy.
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
        model.generation_config.top_k = 50

    model.eval()

    return LocalHFLLM(tokenizer=tokenizer, model=model, max_new_tokens=max_new_tokens)


def parse_args():
    parser = argparse.ArgumentParser(description="Ranking de titulares con LLM local descargado de Hugging Face")
    parser.add_argument(
        "--model",
        default=os.getenv("HF_MODEL", DEFAULT_MODEL),
        help="Ruta local o ID del modelo de Hugging Face.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("HF_CACHE_DIR"),
        help="Directorio local opcional para la cache; si no se indica, se usa la cache por defecto de Hugging Face.",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "4bit", "8bit"],
        default=os.getenv("HF_QUANTIZATION", "none"),
        help="Cuantizacion opcional para modelos grandes.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("HF_MAX_NEW_TOKENS", "2048")),
        help="Numero maximo de tokens de salida por noticia.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Permite codigo remoto del repo del modelo (necesario para algunos modelos).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="No descarga desde Hugging Face; solo usa rutas locales ya existentes.",
    )
    parser.add_argument(
        "--target-column",
        default=os.getenv("HF_TARGET_COLUMN", "y_true"),
        help="Columna con la etiqueta correcta para calcular metricas (por defecto: y_true).",
    )
    return parser.parse_args()


def get_hf_token():
    token = os.getenv("HF_TOKEN", "").strip()
    return token or None


def extract_ranking(response, n_titles=10):
    def normalize(nums):
        seen = set()
        ordered = []
        for n in nums:
            if 1 <= n <= n_titles and n not in seen:
                seen.add(n)
                ordered.append(n)

        if not ordered:
            return None

        for n in range(1, n_titles + 1):
            if n not in seen:
                ordered.append(n)

        return [str(n) for n in ordered]

    # 1) Intento estricto: lista entre corchetes.
    bracket_matches = re.findall(r"\[(.*?)\]", response, flags=re.DOTALL)
    for chunk in bracket_matches:
        nums = [int(x) for x in re.findall(r"\b(?:10|[1-9])\b", chunk)]
        parsed = normalize(nums)
        if parsed is not None:
            return parsed

    # 2) Fallback: numeros validos en cualquier parte de la respuesta.
    nums_anywhere = [int(x) for x in re.findall(r"\b(?:10|[1-9])\b", response)]
    return normalize(nums_anywhere)

INFERENCE = None
MODELO = DEFAULT_MODEL

def predict_headline_ranking(body, titles):
    """
    Solicita al LLM local (Hugging Face) que ordene los 10 titulares por relevancia.
    """
    titulares_formateados = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])

    user_prompt = f"NOTICIA:\n{body[:2000]}\n\nTITULARES POSIBLES:\n{titulares_formateados}"

    try:
        response = INFERENCE.generate(SYSTEM_PROMPT, user_prompt)
        #print(f"\nRespuesta cruda del modelo: {response}\n")

        ranking = extract_ranking(response, n_titles=len(titles))
        if ranking is None:
            print(f"Aviso: no se pudo parsear ranking. Respuesta: {response[:240]}")
        return ranking

    except Exception as e:
        print(f"\nError de inferencia local: {e}")
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
    args = parse_args()
    hf_token = get_hf_token()

    MODELO = args.model
    try:
        INFERENCE = load_local_hf_model(
            model_name_or_path=MODELO,
            quantization=args.quantization,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
            download_from_hub=not args.no_download,
            cache_dir=args.cache_dir,
            token=hf_token,
        )
    except Exception as e:
        raise RuntimeError(
            f"{e}"
        )

    archivo_muestra = os.path.join(DATA_DIR, "test_public.csv")

    if not os.path.exists(archivo_muestra):
        raise FileNotFoundError(f"No se encuentra '{archivo_muestra}'.")

    df_train = pd.read_csv(archivo_muestra)
    sample_df = df_train.copy()

    rankings = []
    print(f"Iniciando procesado de {len(sample_df)} noticias.")
    print(f"Iniciando ranking con modelo Hugging Face: {MODELO} (quantization={args.quantization})...")

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        titles = [row[f'title_{i}'] for i in range(1, 11)]
        body = row['article_body']

        rank = predict_headline_ranking(body, titles)
        rankings.append(rank)

    sample_df['pred_ranking'] = rankings

    if args.target_column in sample_df.columns:
        sample_df['target_num'] = sample_df[args.target_column].astype(str).str.extract(r'(\d+)')

        # Calculo de metricas solo cuando hay etiqueta objetivo.
        sample_df['mrr'] = sample_df.apply(lambda x: calculate_mrr(x['target_num'], x['pred_ranking']), axis=1)
        sample_df['ndcg'] = sample_df.apply(lambda x: calculate_ndcg(x['target_num'], x['pred_ranking']), axis=1)
        sample_df['hit_at_3'] = sample_df.apply(lambda x: 1.0 if str(x['target_num']) in (x['pred_ranking'][:3] if x['pred_ranking'] else []) else 0.0, axis=1)

        print("\n--- Métricas ---")
        print(f"Accuracy (Top-1): { (sample_df['mrr'] == 1.0).mean() * 100:.2f}%")
        print(f"Mean MRR: {sample_df['mrr'].mean():.4f}")
        print(f"Mean NDCG: {sample_df['ndcg'].mean():.4f}")
        print(f"Hit Rate @3: {sample_df['hit_at_3'].mean() * 100:.2f}%")
    else:
        print(
            f"\nNo se encontro la columna objetivo '{args.target_column}'. "
            "Se omiten metricas supervisadas y solo se guardan rankings predichos."
        )

    model_suffix = safe_model_folder_name(MODELO)
    output_file = f"results/resultados_texto_local_{model_suffix}.csv"
    os.makedirs("results", exist_ok=True)
    sample_df.to_csv(output_file, index=False)
    print(f"\nResultados guardados con éxito en '{output_file}'")