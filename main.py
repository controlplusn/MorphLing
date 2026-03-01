from src import MorphlingTokenizer

if __name__ == "__main__":
    s = 'Tumitigil-tigil kayo, at kung hindi ay "pagbabalibagin" ko \'yong lahat.'
    # s = "Ako'y may lobo.\nLumipad nga lang sa langit tangina"
    # s = "Pinaglalabanang"
    tokenizer = MorphlingTokenizer()

    print(tokenizer.tokenize(s))
    tokens = tokenizer.encode(s)
    print(s)
    print(tokens)
    print(tokenizer.decode(tokens))
    # print(tokenizer.get_vocab())
