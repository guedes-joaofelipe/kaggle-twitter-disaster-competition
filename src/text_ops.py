import re
import string

import nltk

from src import constants


def clean_keyword(keyword: str) -> str:
    if not isinstance(keyword, str):
        return ""

    return keyword.replace("%20", " ").lower()


def clean_text(text: str) -> str:
    return text.replace("%20", " ").lower()


def get_text_tags(text: str, regex=r"#") -> list:
    return re.findall(regex, text)


def count_character(text: str, character: str) -> int:
    return len([char for char in str(text) if char == character])


def get_ner_labels(text, nlp, restrictions: list = []) -> dict:
    if text in ["", None]:
        return {}

    doc = nlp(text)
    labels = {
        entity.text: entity.label_
        for entity in doc.ents
        if entity.label_ in restrictions
    }
    if len(restrictions) > 0:
        labels = {
            text: label for text, label in labels.items() if label in restrictions
        }

    return labels


def get_location_labels(text, nlp):
    labels = {}

    candidates = get_ner_labels(text, nlp, restrictions=["GER", "ORG", "LOC"])
    for text, label in candidates.items():
        if text in constants.STATES.keys():
            text = constants.STATES[text]
        labels[text] = label

    return labels


def clean_text(text):
    text = text.lower()

    stop_words = set(nltk.corpus.stopwords.words("english"))
    text = re.sub(f"'[a-z]", "", text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = nltk.tokenize.word_tokenize(text)
    stemmer = nltk.stem.porter.PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    text = " ".join(tokens)

    return text
