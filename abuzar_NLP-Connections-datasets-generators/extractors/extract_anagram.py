import nltk
nltk.download("words")
nltk.download("wordnet")
from nltk.corpus import words, wordnet

MIN_LEN = 4
MAX_LEN = 8



def is_good_anagram_word(word):
    if not word.isalpha():
        return False

    if not word.islower():
        return False

    if not (MIN_LEN <= len(word) <= MAX_LEN):
        return False

    if not wordnet.synsets(word):
        return False

    return True


def main():
    vocab = sorted({word for word in words.words() if is_good_anagram_word(word)})

    with open("anagram_words.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")

    print(f"Saved {len(vocab)} words to anagram_words.txt")


if __name__ == "__main__":
    main()
