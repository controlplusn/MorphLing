from src import MorphlingTokenizer

if __name__ == "__main__":
    s = 'Tumitigil-tigil kayo, at kung hindi ay "pagbabalibagin" ko "kayong" lahat.'
    # s = "aalis"
    tokenizer = MorphlingTokenizer()

    tokens = tokenizer.tokenize(s)
    print(s)
    print(tokens)
    print(tokenizer.detokenize(tokens))
