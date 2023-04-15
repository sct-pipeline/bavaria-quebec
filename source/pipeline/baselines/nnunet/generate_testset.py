import pandas as pd
import random
import json 

df = pd.read_csv("subject_paths.csv", delimiter=";")
x = df["path"].astype(str).values.tolist()

random.seed(42)
random.shuffle(x)

d= {}
d["train"] = sorted(x[:int(0.7*len(x))])
d["valid"] = sorted(x[int(0.7*len(x)):int(0.9*len(x))])
d["test"] = sorted(x[int(0.9*len(x)):])

with open('ivado_splitV2.json', 'w') as fp:
    json.dump(d,fp,indent=4)


