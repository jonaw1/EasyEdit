import os
import json

if os.path.exists("used_case_ids.json"):
    with open("used_case_ids.json", "r") as f:
        used_case_ids = json.load(f)
else:
    used_case_ids = {}

for i in range(130, 160):
    used_case_ids[i] = True

with open("used_case_ids.json", "w") as f:
    json_string = json.dumps(used_case_ids)
    f.write(json_string)