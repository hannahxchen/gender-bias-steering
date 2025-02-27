import os, re
from pathlib import Path
import pandas as pd

# Original dataset can be accessed here: https://arrow.tudublin.ie/datas/24/
DATA_DIR = Path(__file__).resolve().parent / "dataset" / "Dataset-for-Gendered-Language"

def filter_gendered_words(pattern, text):
    filtered = re.sub(pattern, '', text)
    filtered = re.sub(r" 's", '', filtered) # Remove _'s
    filtered = re.sub(r'\s{2,}', ' ', filtered) # Remove extra white space
    return filtered


word_replacements = {
    r'\b(?:she|he) is\b': 'they are',
    r'\b(?:she|he) was\b': 'they were',
    r'\b(?:She|He) is\b': 'They are',
    r'\b(?:She|He) was\b': 'They were',
    r'\b(?:male|female) athlete\b': 'athlete',
    r'\bfemale heroes\b': 'heroes',
    r'\b(?:female pilots|women pilots)\b': 'pilots',
    r'\b(?:she|he)\b': 'they',
    r'\b(?:her|his)\b': 'their',
    r'\b(?:She|He)\b': 'They',
    r'\b(?:Her|His)\b': 'Their',
    r'\bhim\b': 'them',
    r'\b(?:herself|himself)\b': 'themselves',
    r'\b(?:woman|man|lady|gentleman|male|female)\b': 'person',
    r'\b(?:women|men|males|females)\b': 'people',
    r'\b(?:Women|Men)\b': 'People',
    r'\b(?:boy|girl)\b': 'kid',
    r'\b(?:boys|girls)\b': 'kids',
    r'\b(?:Boys|Girls)\b': 'kids',
    r'\b(?:schoolboy|schoolgirl)\b': 'schoolchild',
    r'\b(?:schoolboys|schoolgirls)\b': 'schoolchildren',
    r'\b(?:ladies|gentlemen)\b': 'folks',
    r'\b(?:masculine|feminine|macho)\b': '',
    r'\b(?:masculinity|femininity)\b': '',
    r'\b(?:mrs.|mr.|manly|motherly)\b': '',
    r'\b(?:mother|father|mom|dad)\b': 'parent',
    r'\b(?:mothers|fathers|moms|dads)\b': 'parents',
    r'\b(?:maternal|paternal)\b': 'parental',
    r'\b(?:brother|sister)\b': 'sibling',
    r'\b(?:brothers|sisters)\b': 'siblings',
    r'\b(?:stepmother|stepfather)\b': 'stepparent',
    r'\b(?:mothers|fathers)\b': 'parents',
    r'\b(?:godmother|godfather)\b': 'godparent',
    r'\b(?:grandmother|grandfather|grandma|grandpa)\b': 'grandparent',
    r'\b(?:son|daughter)\b': 'child',
    r'\b(?:sons|daughters)\b': 'children',
    r'\b(?:grandson|granddaughter)\b': 'grandchild',
    r'\b(?:grandsons|granddaughters)\b': 'grandchildren',
    r'\b(?:husband|wife)\b': 'spouse',
    r'\b(?:aunt|uncle)\b': 'relative',
    r'\b(?:boyfriend|girlfriend)\b': 'partner',
    r'\b(?:king|queen)\b': 'ruler',
    r'\b(?:prince|princess)\b': 'noble',
    r'\b(?:nephew|niece)\b': 'cousin',
    r'\b(?:waiter|waitress)\b': 'server',
    r'\b(?:actor|actress)\b': 'cast-member',
    r'\b(?:wizard|witch)\b': 'mage',
    r'\b(?:nun|monk)\b': 'monastic',
    r'\b(?:businessman|businesswoman)\b': 'businessperson',
    r'\bbusinessmen\b': 'businesspeople',
    r'\bgoddess\b': 'god',
    r'\bheroine\b': 'hero',
    r'\bheroines\b': 'heros',
    r'\bsuperheroine\b': 'superhero',
    r'\bballerina\b': 'ballet dancer',
    r'\bpriest\b': 'clergy',
    r'\bpriestly\b': 'clerical',
    r'\bcowboy\b': 'rancher',
    r'\b(?:brotherhood|sisterhood)\b': 'kinship',
}

name_replacements = {
    r'\bBobby\b': 'Robbie',
    r'\bChris\b': 'Kris',
    r'\bDavid\b': 'Devon',
    r'\bEthan\b': 'Emery',
    r'\bJohn\b': 'Johnnie',
    r'\bJack\b': 'Jackie',
    r'\bJames\b': 'Jamie',
    r'\bJim\b': 'Jean',
    r'\bJason\b': 'Jaden',
    r'\bMichael|Mike\b': 'Michel',
    r'\bMatt\b': 'Matty',
    r'\bMark\b': 'Merle',
    r'\b(?:Patrick|Peter)\b': 'Payton',
    r'\bGoliath\b': 'Grey',
    r'\bRobert\b': 'Robin',
    r'\b(?:Tom|Terry)\b': 'Toni',
    r'\b(?:Audrey|Annie)\b': 'Aubrey',
    r'\bCarla\b': 'Casey',
    r'\bCynthia\b': 'Charlie',
    r'\b(?:Emily|Emma)\b': 'Emery',
    r'\bElizabeth\b': 'Emerson',
    r'\bLisa\b': 'Leslie',
    r'\bLucy\b': 'Lou',
    r'\b(?:Lily|Linda)\b': 'Levi',
    r'\bLaura\b': 'Loren',
    r'\bSamantha\b': 'Sam',
    r'\bJessica\b': 'Jessie',
    r'\bJulia\b': 'Jules',
    r'\b(?:Marie|Maria)\b': 'Marley',
    r'\b(?:Mary|Mia)\b': 'Morgan',
    r'\b(?:Melissa|Martha)\b': 'Merritt',
    r'\bNaomi\b': 'Navi',
    r'\bPauline\b': 'Phoenix',
    r'\bOlivia\b': 'Ollie',
    r'\b(?:Jane|Jenny)\b': 'Jayden',
    r'\b(?:Rachel|Ruth)\b': 'Riley',
    r'\b(?:Hannah|Helen)\b': 'Hollis',
    r'\b(?:Susan|Susie|Sarah)\b': 'Sage',
    r'\bSophia\b': 'Sidney',
    r'\bStephanie\b': 'Stevie',
}

def replace_words(x):
    for i, j in word_replacements.items():
        x = re.sub(i, j , x)
    # for n1, n2 in name_replacements.items():
    #     x = re.sub(n1, n2, x)
    x = re.sub(r'\s{2,}', ' ', x) # Remove extra white space
    return x

def process_gendered_language_dataset():
    combined = None
    for i in range(1, 3):
        for prefix in ["Cryan_dataset", "Gaucher_dataset"]:
            df = pd.read_csv(DATA_DIR / f"{prefix}_set_{i}.csv")
            if combined is None:
                combined = df
            else:
                combined = pd.concat((combined, df))

    # combined = combined[combined.Labels == "consistent"]
    combined.loc[(combined.about_Male == 1) & (combined.Masc_terms == 1), 'gender_label'] = "M"
    combined.loc[(combined.about_Male == 0) & (combined.Masc_terms == 0), 'gender_label'] = "F"

    combined.loc[(combined.about_Male == 1) & (combined.Masc_terms == 0), 'gender_label'] = "M_contradict"
    combined.loc[(combined.about_Male == 0) & (combined.Masc_terms == 1), 'gender_label'] = "F_contradict"
    # combined.loc[combined.about_Male != combined.Masc_terms, 'gender_label'] = "N"
    combined = combined.drop(columns=["Index", "about_Male", "Masc_terms"])
    combined = combined[["Sentences", "gender_label"]]
    combined.rename(columns={"Sentences": "text"}, inplace=True)
    combined = combined.reset_index(drop=True)
    combined["_id"] = combined.index
    combined = combined.sample(frac=1, random_state=5678)

    neutral = combined.sample(frac=0.5, random_state=5678)
    neutral["text"] = neutral["text"].apply(lambda x: replace_words(x))
    non_neutral = combined[~combined._id.isin(neutral._id)].copy()
    neutral["is_neutral"] = True
    non_neutral["is_neutral"] = False

    neutral_size = int(len(neutral) / 2)
    non_neutral_size = int(len(non_neutral) / 2)

    train_neutral = neutral[:neutral_size]
    val_neutral = neutral[neutral_size:]
    train = non_neutral[:non_neutral_size]
    val = non_neutral[non_neutral_size:]
    
    train = pd.concat((train, train_neutral))
    val = pd.concat((val, val_neutral))

    os.makedirs(DATA_DIR / "splits", exist_ok=True)
    train.to_csv(DATA_DIR / "splits/train.csv", index=False)
    val.to_csv(DATA_DIR / "splits/val.csv", index=False)


def main():
    process_gendered_language_dataset()


if __name__ == "__main__":
    main()