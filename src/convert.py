import pandas
import sys
import json


files = sys.argv[1:]
for file in files:
    if not file.endswith('.csv'):
        continue
    df = pandas.read_csv(file)
    programs = df["dreamcoder_program"].tolist()
    with open(file[:-4] + '.json', 'w') as f:
        json.dump(programs, f, indent=4)
    print(f"Converted {file} to {file[:-4] + '.json'}")

