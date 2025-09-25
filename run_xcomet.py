import sys
import json
from recognized_lang_pairs import lang_pairs
from comet import download_model, load_from_checkpoint


# Run XCOMET-XL on the prepared input files and print out the score information
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected a single argument 'language pair'.")
    elif sys.argv[1] not in lang_pairs:
        print(f"Error: Argument '{sys.argv[1]}' is not a recognized language pair.")
    else:
        # --- Load and run XCOMET-XL ---
        working_directory = f"language_pairs/{sys.argv[1]}/xcomet-xl"
        comet_inputs = []
        with open(f"{working_directory}/segments.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                comet_inputs.append(json.loads(line))

        model_path = download_model("Unbabel/XCOMET-XL")
        model = load_from_checkpoint(model_path)

        # Adjust batch_size/gpus to your machine (gpus=1 on a CUDA box; set gpus=0 for CPU)
        model_output = model.predict(comet_inputs, batch_size=8, gpus=1)

        # Segment-level score
        print ("Segment-level scores:")
        print (model_output.scores)

        # System-level score
        print ("System-level score:")
        print (model_output.system_score)

        # Score explanation (error spans)
        print ("Score explanation")
        print (model_output.metadata.error_spans)