import os
import json
from pipeline import PersonSearchPipeline


def main():
    # Prompt only for description
    print("=== Multi-Camera Person Search & Tracking ===")
    description = input("Enter a natural language description of the person to search for: ")

    # Use default paths without further prompts
    cams_root = "data/Market1501/cams"
    output_json = "market_output.json"

    # 1) Collect per-camera folder paths
    if not os.path.isdir(cams_root):
        print(f"Error: camera root directory '{cams_root}' does not exist.")
        return
    cams = sorted(os.listdir(cams_root))
    cams = [os.path.join(cams_root, c) for c in cams]

    # 2) Instantiate the pipeline and run search + track
    print("\nRunning search and tracking...")
    ps = PersonSearchPipeline()
    result = ps.search_and_track(description, camera_dirs=cams)

    # 3) Save cropped snapshots for each sighting
    ps.save_snapshots(result)

    # 4) Write out JSON, using default=int to handle numpy int types
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2, default=int)

    # 5) Pretty-print the result to console
    print(f"\nResults saved to {output_json}")
    print(json.dumps(result, indent=2, default=int))


if __name__ == "__main__":
    main()
