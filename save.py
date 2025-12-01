import json
import os

def save_to_json(query, answer, json_file="visa_data.json"):
    # Load existing JSON or create new
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Add/Update query-answer pair
    data[query] = answer

    # Save back to file
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

    print("[INFO] Data saved to visa_data.json")
