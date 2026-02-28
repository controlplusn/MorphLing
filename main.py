from src import MorphlingTokenizer

if __name__ == "__main__":
    # s = 'Tumitigil-tigil kayo, at kung hindi ay "pagbabalibagin" ko \'yong lahat.'
    s = "Ako'y may lobo.\nLumipad nga lang sa langit tangina"
    tokenizer = MorphlingTokenizer()

    tokens = tokenizer.tokenize(s)
    print(s)
    print(tokens)
    print(tokenizer.detokenize(tokens))
