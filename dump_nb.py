import json

nb_path = "HindiTranslator_fixed.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    
    if "chunk_slider =" in src:
        print(f"--- CELL {i} (SLIDER definitions) ---")
        lines = src.split('\n')
        for i_l, l in enumerate(lines):
            if "slider" in l or "Layout" in l:
                print(f"{i_l}: {l}")

    if "SYSTEM_PROMPT" in src:
        # Cell 14
        print(f"\n--- CELL {i} (ENGINE code parts) ---")
        
        # We need to find SYSTEM_PROMPT
        start_sys = src.find("SYSTEM_PROMPT =")
        end_sys = src.find('"""\n', start_sys + 10) + 4
        print("SYSTEM_PROMPT (first 200 chars):", repr(src[start_sys:start_sys+200]))
        print("SYSTEM_PROMPT ends at:", repr(src[end_sys-20:end_sys+10]))

        # We need the end of alpha_density
        start_alpha = src.find("def alpha_density")
        end_alpha = src.find("def", start_alpha + 1)
        # Or look before # -- Prompts --
        prompts_idx = src.find("# -- Prompts --")
        print("Lines before '# -- Prompts --':")
        print(repr(src[prompts_idx-100:prompts_idx+20]))

        # After alpha_density check around line 749 in prompt, but let's look for alpha_density in translate_chunk
        alpha_call = src.find("alpha_density(raw)")
        print("Around alpha_density(raw) call:")
        print(repr(src[alpha_call-50:alpha_call+150]))

        # Options dictionary
        options_idx = src.find("options={")
        options_end = src.find("}", options_idx) + 1
        print("Options dict:\n", repr(src[options_idx:options_end]))

        # Preamble stripper
        re_sub_idx = src.find("raw = re.sub(")
        re_sub_end = src.find("raw = parts", re_sub_idx) # wait, prompt says "Replace the re.sub that strips preamble labels"
        if re_sub_end == -1: re_sub_end = src.find(".strip()", re_sub_idx) + 8
        else: re_sub_end += 11 # rough guess
        print("First re.sub block:\n", repr(src[re_sub_idx:re_sub_end+50]))
        
        # maybe there's a second one since there's multiple re.subs? Let's just find "preamble labels"
        preamble_comment = src.find("preamble labels")
        print("Around 'preamble labels':\n", repr(src[preamble_comment-50:preamble_comment+300]))

    if "Good Examples" in src:
        print(f"\n--- CELL {i} (MARKDOWN Table) ---")
        start_ge = src.find("### ")
        print(repr(src[start_ge:]))

