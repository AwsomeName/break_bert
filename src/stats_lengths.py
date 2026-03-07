import json
import numpy as np

def main():
    with open("data/raw/samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    lengths = []
    for s in samples:
        full_text = "\n".join(s)
        lengths.append(len(full_text))
    
    stats = {
        "count": len(lengths),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "avg": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "individual_lengths": lengths
    }
    
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
