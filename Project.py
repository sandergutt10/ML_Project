import json
from collections import Counter

############################################
# 1. Загрузка и токенизация AST
############################################

def load_ast_dataset(path):

    tokens = []

    with open(path, "r", encoding="utf8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            ast_nodes = json.loads(line)

            for node in ast_nodes:

                tokens.append("TYPE_" + node["type"])

                if "value" in node:
                    tokens.append("VALUE_" + str(node["value"]))

    return tokens

############################################
# 2. Построение словаря
############################################

def build_vocab(tokens):

    counter = Counter(tokens)

    vocab = {tok: i for i, tok in enumerate(counter.keys())}
    ivocab = {i: tok for tok, i in vocab.items()}

    return vocab, ivocab


############################################
# 3. Кодирование токенов
############################################

def encode(tokens, vocab):

    return [vocab[t] for t in tokens if t in vocab]
