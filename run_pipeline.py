import json
from pipeline import PersonSearchPipeline

if __name__ == "__main__":
    desc = input("Enter person description: ")
    ps = PersonSearchPipeline()
    result = ps.search_and_track(desc)
    ps.save_snapshots(result)
    with open("output.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved JSON → output.json")
