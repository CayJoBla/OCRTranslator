import requests

DESERET_RESOURCE_URL = "http://2deseret.com/json/translation"
STEEL_SPECIAL_WORDS = [
    "iron", "steel", "tin", "pewter", "zinc", "brass", "copper", "bronze", 
    "cadmium", "bendalloy", "gold", "electrum", "chromium", "nicrosil", 
    "aluminum", "aluminium", "duralumin", "duraluminium", "atium", "malatium", 
    "lerasium", "ettmetal", "harmonium", "west", "northwest", "north", 
    "northeast", "east", "southeast", "south", "southwest"
]

def english_to_ipa(text):
    raise NotImplementedError("This feature is not yet implemented")

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
    text = text.lower()

    # TODO: Check which symbols are not accounted for in our font

    # TODO: Changes to apply
    # Writing changes:
    #     Account for double characters (tt -> t, ll -> l)
    #     Account for ie, ei, ou, etc. (should be a single symbol)
    #     Capitalize special words (IRON, STEEL, NORTH, SOUTH, etc.)
    #           (Note: These are rendered differently by the font)
    # Pronunciation changes:
    #     Account for silent letters (knight -> nait)
    #     Account for special pronunciation (sh, ch, ph, gh)
    #     Determine pronunciation for c (hard c -> k, soft c -> s)

    # TODO: Should I implement a proper number translator? 
    #       The font number translator would need to be disabled since it breaks
    #       with certain numbers (167 -> 16 and 7)

    raise NotImplementedError("This feature is not yet implemented")

def steel_phonetics_to_english(text):
    raise NotImplementedError("Good luck bud.")

if __name__ == "__main__":
    text = input("Enter text to translate: ")
    print(translate(text))