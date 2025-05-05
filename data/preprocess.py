import pandas as pd
import re

contractions = {
    "'re": " are",
    "n't": " not",
    "'s": " is",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
    "'ve": " have",
    "'em": " them",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "mightn't": "might not",
    "mustn't": "must not",
    "wasn't": "was not",
    "weren't": "were not",
    "isn't": "is not",
    "aren't": "are not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
}

def expand_contractions(text, contractions_dict):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in contractions_dict.keys()) + r')\b')
    return pattern.sub(lambda x: contractions_dict[x.group()], text)

def preprocessing(data):
    data = data.str.normalize('NFD')
    data = data.str.encode('ascii', 'ignore').str.decode('utf-8')
    data = data.str.lower()
    data = data.str.replace(r"([^ a-z.?!¡,¿'-])", "", regex=True)
    data = data.str.replace(r"([?.!¡,¿])", r" \1 ", regex=True)
    data = data.str.replace(r'[" "]+', " ")
    data = data.apply(lambda x: expand_contractions(x, contractions))
    data = data.str.strip()
    return data

if __name__ == '__main__':
    df = pd.read_csv('data/raw/ind-eng/ind.txt', sep='\t', header=None)
    df.columns = ['en', 'id', 'cc']
    df.drop(columns=['cc'], inplace=True)
    print("Data sebelum preprocessing:")
    print(df.head())
    df['en'] = preprocessing(df['en'])
    df['id'] = preprocessing(df['id'])
    print("\nData setelah preprocessing:")
    for i, target in enumerate(df['en'][-10:].values):
        print(f"{i}: {target}")
    df.to_csv('data/processed/id-en.csv', index=False)
    