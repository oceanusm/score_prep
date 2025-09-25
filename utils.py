
import os
import json
import pandas as pd

def _get_ref_text(refs):
    if isinstance(refs, dict) and "refA" in refs and isinstance(refs["refA"], dict):
        return refs["refA"].get("ref", "")
    return ""


def _split_paragraphs(text: str):
    # Do not strip aggressively; preserve internal content, only normalize trailing whitespace.
    return [p.rstrip() for p in text.split("\n\n")]


def combine_src_hyp_ref(hyp_path):
    SRC_REF_PATH = 'wmt25-genmt.jsonl'

    # --- Load source & reference into a doc_id-indexed frame ---
    raw_df = pd.read_json(SRC_REF_PATH, lines=True)
    # Put refA into ref_text column, or put empty string if reference doesn't exist. 
    raw_df["ref_text"] = raw_df["refs"].apply(_get_ref_text)
    # New df containing doc_id, src text, and ref text (or empty string)
    df = raw_df.set_index("doc_id")[["src_text", "ref_text"]]

    # --- Match hypotheses to the corresponding src and ref texts ---
    segments = []
    skipped_failed = 0
    total_hyps = 0

    with open(hyp_path, "r", encoding="utf-8") as f:
        for line in f:
            # Get the doc_id and hypothesis
            rec = json.loads(line)
            total_hyps += 1

            doc_id = rec["doc_id"]
            hyp = rec.get("hypothesis", "")
            # If the translation failed, skip
            if hyp.startswith("FAILED"):
                skipped_failed += 1
                continue
            
            # Lookup source & reference by doc_id
            src = df.at[doc_id, "src_text"]
            ref = df.at[doc_id, "ref_text"]
            
            # Split src, hyp, and ref into paragraphs
            src_pars = _split_paragraphs(src)
            hyp_pars = _split_paragraphs(hyp)
            ref_pars = _split_paragraphs(ref) if ref else [""] * len(src_pars)

            # Build the raw segment pairs
            for i, (s, h, r) in enumerate(zip(src_pars, hyp_pars, ref_pars)):
                segments.append({
                    "doc_id": doc_id,
                    "par_ind": i,
                    "src": s,
                    "hyp": h,
                    "ref": r,
                })

    print(f"Total hypothesis lines: {total_hyps}")
    print(f"Built segments: {len(segments)}  | skipped FAILED: {skipped_failed}")

    return segments


# --- Given a list of raw segments, save them in the appropriate format at the ---
# Save the raw segments.
def save_raw_segments(segments):
    # Make sure segments isn't empty
    if len(segments) == 0:
        print("List 'segments' is empty. Did not save.")
        return
    
    # Get the language pair string so we know where to write to.
    lang_pair_str = segments[0]["doc_id"].split("_#")[0]
    output_dir = f"language_pairs/{lang_pair_str}/raw"

    # Save the files
    print(f"Saving to {output_dir}/segments.jsonl")
    os.makedirs(output_dir, exist_ok=True)  
    with open(f"{output_dir}/segments.jsonl", "w", encoding="utf-8") as f:  
        for seg in segments:  
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")  


# xCOMET-XL can be run with Python code or from CLI. CLI use is unclear, so run via Python. We can read in a jsonl of form "src", "mt", "ref". 
# If no reference, "ref" field needs to be ommitted.
def save_segments_for_xcomet(segments):
    # Make sure segments isn't empty
    if len(segments) == 0:
        print("List 'segments' is empty. Did not save.")
        return
    
    # Get the language pair string so we know where to write to.
    lang_pair_str = segments[0]["doc_id"].split("_#")[0]
    output_dir = f"language_pairs/{lang_pair_str}/xcomet-xl"

    # Process the raw segments appropriately
    include_ref = True
    if segments[0]["ref"] == "":
        include_ref = False

    formatted_segments = []
    for seg in segments:
        # Alwyas need src and mt
        new_seg = {
            "src": seg["src"],
            "mt": seg["hyp"]
        }

        # Include ref too if it isn't ""
        if include_ref:
            new_seg["ref"] = seg["ref"]

        formatted_segments.append(new_seg)

    # Save the files
    print(f"Saving to {output_dir}/segments.jsonl")
    os.makedirs(output_dir, exist_ok=True)  
    with open(f"{output_dir}/segments.jsonl", "w", encoding="utf-8") as f:  
        for rec in formatted_segments:  
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")  


# MetricX-24 expects input.jsonl with "source", "hypothesis", and "reference" fields per line. 
# If no reference, "reference" = "".
def save_segments_for_metricx24(segments):
    # Make sure segments isn't empty
    if len(segments) == 0:
        print("List 'segments' is empty. Did not save.")
        return
    
    # Get the language pair string so we know where to write to.
    lang_pair_str = segments[0]["doc_id"].split("_#")[0]
    output_dir = f"language_pairs/{lang_pair_str}/metricx-24"

    # Process the raw segments appropriately
    formatted_segments = []
    for seg in segments:
        formatted_segments.append(
            {
                "source": seg["src"],
                "hypothesis": seg["hyp"],
                "reference": seg["ref"]
            }
        )


    # Save the files
    print(f"Saving to {output_dir}/segments.jsonl")
    os.makedirs(output_dir, exist_ok=True)  
    with open(f"{output_dir}/segments.jsonl", "w", encoding="utf-8") as f:  
        for rec in formatted_segments:  
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")  




# GEMBA requires two parallel (line by line) txt files containing the source and the hypothesis. 
# TODO NOTE: We normalize by replacing newlines in the segments with spaces, but this wasn't explicitly described as necessary. 
def save_segments_for_gemba(segments):
    # Make sure segments isn't empty
    if len(segments) == 0:
        print("List 'segments' is empty. Did not save.")
        return
    
    # Get the language pair string so we know where to write to.
    lang_pair_str = segments[0]["doc_id"].split("_#")[0]
    output_dir = f"language_pairs/{lang_pair_str}/gemba"

    # Process the raw segments appropriately
    src_segs = []
    hyp_segs = []

    for seg in segments:
        src_segs.append(seg["src"].replace("\n", " ")) # Replace \n chars
        hyp_segs.append(seg["hyp"].replace("\n", " ")) # Replace \n chars

    # Save the files
    print(f"Saving to {output_dir}/src.txt")
    print(f"Saving to {output_dir}/hyp.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/src.txt", "w", encoding="utf-8") as f_src:  
        f_src.write("\n".join(src_segs))  
    with open(f"{output_dir}/hyp.txt", "w", encoding="utf-8") as f_hyp:  
        f_hyp.write("\n".join(hyp_segs))  