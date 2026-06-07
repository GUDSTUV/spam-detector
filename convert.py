import pandas as pd

df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["v1", "v2"])
df.to_csv("spam.csv", index=False)
print("Done!", len(df), "messages saved to spam.csv")