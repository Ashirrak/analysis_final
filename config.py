"""Configuration constants for the genomic simulation analysis suite."""

CONDITIONS_MAP = {
    "C1": {"mutation_rate": 2.5e-5, "recomb_rate": 0.005},
    "C2": {"mutation_rate": 2.5e-5, "recomb_rate": 0.01},
    "C3": {"mutation_rate": 2.5e-5, "recomb_rate": 0.02},
    "C4": {"mutation_rate": 1e-4, "recomb_rate": 0.005},
    "C5": {"mutation_rate": 1e-4, "recomb_rate": 0.01},
    "C6": {"mutation_rate": 1e-4, "recomb_rate": 0.02},
    "C7": {"mutation_rate": 2e-4, "recomb_rate": 0.005},
    "C8": {"mutation_rate": 2e-4, "recomb_rate": 0.01},
    "C9": {"mutation_rate": 2e-4, "recomb_rate": 0.02},
}

MODELS = ["DT", "LR", "NN"]
RESULTS_DIR = "results"
ORIGINAL_DIR = "original_data"

# Tag-level columns from results dataset
TAG_LEVEL_COLUMNS = [
    "Tag", "Tag_RDP_Count", "Tag_Santa_Count", 
    "Tag_Original_RDP_Count", "Tag_Original_Santa_Count",
    "Tag_Remaining_RDP", "Tag_Remaining_Santa",
    "Tag_RDP_Match_Rate", "Tag_Santa_Match_Rate",
    "Tag_Total_Original", "Tag_Total_Matched", "Tag_Total_Remaining",
    "RDP_Santa_Equal"
]

