import json
import pandas as pd

data = []
for line in open(r'C:\Users\Zehra\Desktop\ai\Musical_Instruments_5.json') :
    data.append(json.loads(line))

#print(data[0])
df = pd.DataFrame(data)
#print(len(data))
df.head(10)
df = df.drop(columns=['reviewerName'])
df1 = df.rename(columns = {'overall': 'rating', 'asin': 'productID'}, inplace = False)
df1.dropna(axis = 0, how ='any',inplace=True) 
df1.drop_duplicates(subset=['rating','reviewText'],keep='first',inplace=True)
print(len(df1))
df1.to_csv('Musical_Instruments.csv')