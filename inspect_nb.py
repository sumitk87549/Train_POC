import json
import re

nb_path = "HindiTranslator_fixed.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    info = []
    if "SYSTEM_PROMPT" in src: info.append("SYSTEM_PROMPT")
    if "def repetition_score" in src: info.append("repetition_score")
    if "def alpha_density" in src: info.append("alpha_density")
    if "options={" in src: info.append("options={")
    if "chunk_slider =" in src: info.append("chunk_slider")
    if "re.sub" in src: info.append("re.sub")
    if "Good Examples" in src: info.append("Good Examples")
    
    if info:
        print(f"Cell {i} ({cell.get('cell_type')}): {', '.join(info)}")
