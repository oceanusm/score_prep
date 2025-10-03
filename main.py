from utils import combine_src_hyp_ref, save_raw_segments, save_segments_for_xcomet, save_segments_for_metricx24, save_segments_for_gemba

HYP_PATH = "OPUS-en-ja.jsonl"

# Given a specified file containing hypotheses (output from the collect translations pipeline), 
# Prepare the necessary files for scoring translation performance.
def prepare_score_inputs():
    raw_segments = combine_src_hyp_ref(HYP_PATH)
    save_raw_segments(raw_segments)
    save_segments_for_xcomet(raw_segments)
    save_segments_for_metricx24(raw_segments)
    save_segments_for_gemba(raw_segments)


# Prepare the inputs for scoring.
if __name__ == "__main__":
    prepare_score_inputs()