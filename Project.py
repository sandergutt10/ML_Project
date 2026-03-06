import json

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