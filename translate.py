import requests
import re
import json

DESERET_RESOURCE_URL = "http://2deseret.com/json/translation"


class EnglishToSteelPhonetics:
    def __init__(self, special_words_file="steel_words.json"):
        # Load the special words (words with a special symbol in the steel alphabet)
        self._special_words = {}
        if special_words_file is not None:
            with open(special_words_file) as f:
                self._special_words = json.load(f)

    def _convert_word(self, word):
        transliteration = ""
        syllables = re.split(r"([aeiouy]+)", word)
        for i, syllable in enumerate(syllables): # Split into vowel and consonant groups
            k = 0   # Character index within the current syllable
            while k < len(syllable):
                is_last_char = (k == len(syllable)-1)
                char = syllable[k]
                if char == "c":
                    if not is_last_char and syllable[k+1] == "h":
                        transliteration += "c"      # "ch" sound
                        k += 1  # Skip the "h"
                    elif is_last_char and i < len(syllables) - 1 and syllables[i+1][0] in ["e", "i", "y"]:
                        transliteration += "s"
                    else:
                        transliteration += "k"
                elif char == "p" and not is_last_char and syllable[k+1] == "h":
                    transliteration += "f"
                    k += 1      # Skip the "h"
                elif char == "s" and not is_last_char and syllable[k+1] == "h":
                    transliteration += "x"
                    k += 1      # Skip the "h"
                else:
                    transliteration += char
                k += 1
        return self._post_process(transliteration)

    def _post_process(self, transliteration):
        """No two consecutive characters should be the same, and similar vowels
        should be combined.
        """
        processed = ""
        i = 0
        while i < len(transliteration):
            char = transliteration[i]
            not_last_char = (i < len(transliteration) - 1)
            if not_last_char:
                if char == transliteration[i+1]:
                    i += 1  # Skip the next character (repeated)
                elif char in ("e","i") and transliteration[i+1] in ("e","i"):
                    i += 1  # Skip the next character (similar vowels)
                elif char in ("o","u") and transliteration[i+1] in ("o","u"):
                    i += 1  # Skip the next character (similar vowels)
            processed += char
            i += 1
        return processed

    def _check_punctuation(self, word):
        """Returns True if the word is all punctuation and/or spaces."""
        for char in word:
            if char.isalnum():
                return False
        return True
    
    def convert(self, text):
        """Converts English text to Steel phonetics."""
        # Divide the input text into words
        words = re.findall(r"\b\w+(?:'(?:t|s|re|m|ve|d|ll))?\b|[^\w]+", text)
        transliteration = ""
        for word in words:
            lowercase_word = word.lower()   # Capitalization des not matter in our font
            if lowercase_word in self._special_words:       # Word has a special symbol
                transliteration += self._special_words[lowercase_word]
            elif self._check_punctuation(lowercase_word):   # Word is all punctuation/spaces
                transliteration += word 
            else:                                           # Word is normal
                transliteration += self._convert_word(lowercase_word)
        return transliteration.strip()

def english_to_deseret(text, local=False):
    if local:
        try:
            from deseret_alphabet_translator import EnglishToDeseret
            translation = EnglishToDeseret().translate(text)
        except ImportError:
            raise Exception("Local translation failed. Please install the deseret_alphabet_translator package.")
    else:
        response = requests.post(DESERET_RESOURCE_URL, json={"english": text})
        if response.status_code != 200 or "deseret" not in response.json():
            raise Exception("Error occurred while translating text: " + response.text)
        translation = response.json()["deseret"]
    return translation

def deseret_to_english(text, local=False):
    if local:
        raise NotImplementedError("Local translation is not available.")
    else:
        response = requests.post(DESERET_RESOURCE_URL, json={"deseret": text})
        if response.status_code != 200 or "english" not in response.json():
            raise Exception("Error occurred while translating text: " + response.text)
        translation = response.json()["english"]
    return translation

def english_to_steel_phonetics(text):
    # TODO: Should I implement a proper number translator? 
    #       The font number translator would need to be disabled since it breaks
    #       with certain numbers (167 -> 16 and 7)
    return EnglishToSteelPhonetics().convert(text)

def steel_phonetics_to_english(text):
    raise NotImplementedError("Good luck bud.")

if __name__ == "__main__":
    text = input("Enter text to translate: ")
    print(translate(text))