from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_FOLDERS = [
	Path("results/gemma-gemma"),
	Path("results/mistral-gemma"),
	Path("results/phi-gemma"),
]


def normalize_ranking(value: object) -> str:
	if isinstance(value, str):
		value = value.strip()
		if value:
			try:
				parsed = ast.literal_eval(value)
			except (SyntaxError, ValueError) as exc:
				raise ValueError(f"Invalid ranking value: {value!r}") from exc
		else:
			parsed = []
	else:
		parsed = value

	if not isinstance(parsed, (list, tuple)):
		raise ValueError(f"Ranking must be a list or tuple, got {type(parsed).__name__}")

	tokens: list[str] = []
	for item in parsed:
		if isinstance(item, str):
			item = item.strip()
			if item.startswith("t") and item[1:].isdigit():
				tokens.append(item)
				continue
			if item.isdigit():
				tokens.append(f"t{item}")
				continue
		elif isinstance(item, int):
			tokens.append(f"t{item}")
			continue

		raise ValueError(f"Unsupported ranking item: {item!r}")

	if len(tokens) != 10:
		raise ValueError(f"Ranking must contain exactly 10 tokens, got {len(tokens)}")

	if len(set(tokens)) != 10:
		raise ValueError(f"Ranking contains duplicate tokens: {tokens!r}")

	return " ".join(tokens)


def find_source_file(folder: Path, pattern: str) -> Path:
	matches = sorted(folder.glob(pattern))
	if len(matches) != 1:
		raise FileNotFoundError(
			f"Expected exactly one file matching {pattern!r} in {folder}, found {len(matches)}"
		)
	return matches[0]


def build_results(folder: Path) -> Path:
	text_file = find_source_file(folder, "resultados_texto_local_*.csv")
	multimodal_file = find_source_file(folder, "resultados_multimodal_local_*.csv")

	text_df = pd.read_csv(text_file, usecols=["id", "pred_ranking"])
	multimodal_df = pd.read_csv(multimodal_file, usecols=["id", "pred_ranking"])

	text_df = text_df.rename(columns={"pred_ranking": "task_1"})
	multimodal_df = multimodal_df.rename(columns={"pred_ranking": "task_2"})

	merged = text_df.merge(multimodal_df, on="id", how="inner", validate="one_to_one")

	if len(merged) != len(text_df) or len(merged) != len(multimodal_df):
		raise ValueError(
			f"ID mismatch in {folder}: text={len(text_df)}, multimodal={len(multimodal_df)}, merged={len(merged)}"
		)

	merged["task_1"] = merged["task_1"].map(normalize_ranking)
	merged["task_2"] = merged["task_2"].map(normalize_ranking)

	output_path = folder / "results.csv"
	merged[["id", "task_1", "task_2"]].to_csv(output_path, index=False)
	return output_path


def iter_folders(folder_args: Iterable[str]) -> list[Path]:
	folders = [Path(arg) for arg in folder_args]
	if folders:
		return folders
	return DEFAULT_FOLDERS


def main() -> None:
	parser = argparse.ArgumentParser(description="Build submission CSVs from local ranking outputs.")
	parser.add_argument(
		"folders",
		nargs="*",
		help="Folders containing resultados_texto_local_*.csv and resultados_multimodal_local_*.csv",
	)
	args = parser.parse_args()

	for folder in iter_folders(args.folders):
		output_path = build_results(folder)
		print(f"Wrote {output_path}")


if __name__ == "__main__":
	main()
