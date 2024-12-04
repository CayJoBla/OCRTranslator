# -*- coding: utf-8 -*-
import json
import re
import sys
import logging
from importlib.resources import files

from . import htmlentitydefs
from . import deseret_strings

ROOT_DATA_DIR = files("deseret_alphabet_translator.data")

# NOTE: THIS DOES NOT WORK, IT DOES NOT PASS THE UNIT TESTS

class DeseretToEnglish:

    pron_dict = {}
    deseret_to_ipa = {}
    ipa_to_deseret = {}
    gsl_word_rankings_dict = {}
    ipa_to_phonetic = {}
    ascii_to_deseret = {}

    @staticmethod
    def unescape(text):
        def fixup(m):
            anEntity = m.group(0)
            if text[:2] == "&#":
                # character reference
                try:
                    if text[:3] == "&#x":
                        unicode = unichr(int(anEntity[3:-1], 16))
                        return unicode
                    else:
                        unicode = unichr(int(anEntity[2:-1]))
                        return unicode
                except ValueError as error:
                    pass
            else:
                # named entity
                try:
                    unicode = unichr(htmlentitydefs.name2codepoint[anEntity[1:-1]])
                    return unicode
                except KeyError:
                    pass
            return anEntity # leave as is
        return re.sub(r"&#?\w+;", fixup, text)

    def __init__(self):

        if sys.maxunicode < 1114111:
            raise UnicodeError("Requires Python with UCS-4 support.")

        # ipa_to_deseret maps a deseret unicode character to a list of matching IPA characters
        file = ROOT_DATA_DIR.joinpath("ipa_to_deseret.json")
        with file.open("r") as json_file:
            ipa_to_deseret = json.load(json_file)
            for ipa_char, unicode_id in ipa_to_deseret.items():
                unicode_value = chr(int(unicode_id))
                ipa_chars = self.deseret_to_ipa.get(unicode_value, [])
                ipa_chars.append(ipa_char)
                self.deseret_to_ipa[unicode_value] = ipa_chars
                self.ipa_to_deseret[ipa_char] = unicode_value
            json_file.close()

            # IPA doesn't distinguish between &#66562; ("a" in "far") and &#66569; ("o" in "bob"),
            # so the JSON map has two entries with the same key and we lose the mapping from &#66562; to IPA "A"
            self.deseret_to_ipa[chr(66562)] = [u'A']
            self.ipa_to_deseret[u'A'] = chr(66562)         # Use &#66562; instead of &#66569;

            # Add support for phonetic translation of "ew" ["YU"] (&#66599); but use "ye""oo" (&#66565;) when going from English > Deseret
            self.deseret_to_ipa[chr(66599)] = [u'ju']
            # Add support for phonetic translation of "oi" ["OY"] (&#66598); but use "o","ee" when going from English > Deseret
            self.deseret_to_ipa[chr(66598)] = [u'Oi']


        logging.info( "deseret_to_ipa:")
        for deseret_char, ipa_charset in self.deseret_to_ipa.items():
            logging.info("'%s': '%s'" % (deseret_char.encode("utf-8"), ipa_charset))

        file = ROOT_DATA_DIR.joinpath("deseret_names.json")
        with file.open("r") as json_file:
            deseret_names = json.load(json_file)
            for unicode_id, info in deseret_names.items():
                if info['ascii']:
                    self.ascii_to_deseret[info['ascii'][0]] = chr(int(unicode_id)).lower()
                    self.ascii_to_deseret[info['ascii'][1]] = chr(int(unicode_id))
            json_file.close()

        file = ROOT_DATA_DIR.joinpath("ipa_to_phonetic.json")
        with file.open("r") as json_file:
            self.ipa_to_phonetic = json.load(json_file)

        logging.info(f"ipa_to_phonetic: {self.ipa_to_phonetic}")

        # use GSL word rankings from http://jbauman.com/gsl.html to choose most likely word match
        file = ROOT_DATA_DIR.joinpath("gsl.dat")
        with file.open("r") as gsl_file:
            for line in gsl_file:
                entry = line.split(" ")
                # add 1 because if it's in the GSL it's better than not in the GSL, yet some GSL entries have 0s
                self.gsl_word_rankings_dict[entry[2].strip()] = int(entry[1].strip()) + 1
            gsl_file.close()

        for pron_filename in ["mobypron.unc", "custompron.unc"]:
            file = ROOT_DATA_DIR.joinpath(pron_filename)
            with file.open("r") as pron_file:
                for line in pron_file:
                    entry = line.split(" ")
                    english_word = " ".join(entry[:-1])
                    if "/" in english_word:
                        english_word = english_word.split('/', 1)[0]
                    ipa_word = entry[-1].strip()
                    ipa_word = self.normalize_ipa(ipa_word, english_word=english_word)

                    # If there is a collision, favor lower-case words, in order to avoid using
                    # proper names when there are other words that match.
                    collision = self.pron_dict.get(ipa_word, None)
                    if not collision:
                        self.pron_dict[ipa_word] = english_word
                    else:
                        #print "Found collision between '%s' and '%s'" % (english_word, collision)
                        if collision[0].isupper():
                            # favor lower-case words to avoid using proper names where other words match
                            self.pron_dict[ipa_word] = english_word
                        else:
                            collision_rank = self.gsl_word_rankings_dict.get(collision, 0)
                            cur_word_rank = self.gsl_word_rankings_dict.get(english_word, 0)
                            if cur_word_rank > collision_rank:
                                logging.info("Word '%s' has higher rank than '%s'" % (english_word, collision))
                                self.pron_dict[ipa_word] = english_word

            pron_file.close()

        logging.info("Finished initializing DeseretToEnglish class.")

    def translate(self, deseret):

        deseret = self.replace_ascii_chars_with_deseret_unicode(deseret)

        translation = ''
        for deseret_word in re.findall(r'([^\W\d]+|[\W\d]+)', deseret, re.DOTALL | re.UNICODE):
            translation += self.translate_word(deseret_word)

        return translation.strip()

    def translate_word(self, source_word):

        if source_word is None or re.match(r'^[\W]+$', source_word, re.DOTALL | re.UNICODE):
            #logging.warning("Skipping word '%s'" % ('[None]' if source_word is None else source_word.encode('utf-8')))
            return source_word

        deseret_word = source_word.upper()
        translation = deseret_word

        #logging.warning("Translating '%s'..." % deseret_word.encode('utf-8'))

        ipa_word = self.get_ipa_word(deseret_word)
        if not ipa_word:
            #logging.error("ERROR: Could not get IPA word for %s" % deseret_word)
            pass
        else:
            #logging.info("...IPA word: '%s'" % ipa_word.encode('utf-8'))
            english_word = self.get_english_word(ipa_word)
            if english_word:
                translation = english_word
                #logging.info("...Translation: '%s'" % english_word.encode('utf-8'))
            else:
                translation = self.get_ipa_word(deseret_word, dividers=True, english_phonetic=True)
                #logging.info("No translation found for '%s'; returning phonetic '%s'" % (deseret_word.encode('utf-8'), translation.encode('utf-8')))

        return deseret_strings.apply_capitalization(source_word, translation)

    def ipa_char_for_deseret_char(self, deseret_char):
        ipa_char = None
        ipa_charset = self.deseret_to_ipa.get(deseret_char, None)
        if not ipa_charset:
            logging.error("INTERNAL ERROR: No IPA char equivalent found for deseret char '%s'" % deseret_char.encode('utf-8'))
        else:
            ipa_char = ipa_charset[0]

        return ipa_char

    def normalize_ipa(self, ipa_word, english_word=None):
        """For each character in the IPA word, replace it with the first in the set of matches
        for its corresponding deseret character"""

        if not ipa_word:
            return None

        normal_ipa_word = ''

        # remove stress markings, commas
        ipa_word = re.sub(r"[\',]", '', ipa_word)

        # strip leading and trailing syllable dividers
        if ipa_word[0] == '/':
            ipa_word = ipa_word[1:]
        if ipa_word[-1] == '/':
            ipa_word = ipa_word[:-1]

        if not ipa_word:
            logging.error("INTERNAL ERROR: Nothing left in IPA word '%s' after removing stress markings and leading/trailing syllable dividers." % ipa_word)
            return None

        for ipa_syllable in ipa_word.split('/'):
            if not ipa_syllable:
                continue        # empty syllable
            normal_ipa_syllable = self.normalized_ipa_syllable(ipa_syllable)
            if not normal_ipa_syllable:
                logging.error("INTERNAL ERROR: Failure to normalize IPA syllable '%s' in '%s' ('%s')" % (ipa_syllable, ipa_word, english_word))
                normal_ipa_word += ipa_syllable
            else:
                normal_ipa_word += self.normalized_ipa_syllable(ipa_syllable)

        return normal_ipa_word

    def ipa_syllable_within_syllable(self, ipa_syllable):

        symbols = {
            '(@)': '(@)',
            'ou': 'AU',
            'oi': 'Oi',
            r'(/@/)': '@'
        }

        for chars, ipa_symbol in symbols.items():
            # if ipa_syllable[:2] == "ou":
            #     print "FOUND syllable with 'ou': '%s'" % ipa_syllable

            if ipa_syllable[:len(chars)] == chars:
                # print "Found symbol '%s' inside IPA syllable '%s'" % (chars, ipa_syllable)
                normalized_ipa_syllable = self.ipa_char_for_deseret_char(self.ipa_to_deseret.get(ipa_symbol))
                # print "...Normalized to '%s' via '%s'" % (normalized_ipa_syllable, ipa_symbol)
                return normalized_ipa_syllable, ipa_syllable[len(chars):]

        return None, ipa_syllable

    def normalized_ipa_syllable(self, ipa_syllable):
        normalized_ipa_syllable = None
        deseret_char = self.ipa_to_deseret.get(ipa_syllable, None)
        if not deseret_char:
            if len(ipa_syllable) > 1:
                #print "Checking for symbols within IPA syllable '%s'..." % ipa_syllable
                normalized_ipa_syllable, remaining_chars = self.ipa_syllable_within_syllable(ipa_syllable)
                if normalized_ipa_syllable:
                    if remaining_chars:
                        normalized_ipa_syllable += self.normalized_ipa_syllable(remaining_chars)
                else:
                    # current letter is not part of a multi-letter symbol; process it and move on
                    # to the next letter.
                    normalized_ipa_char = self.normalized_ipa_syllable(ipa_syllable[0])
                    if normalized_ipa_char is not None:
                        normalized_ipa_syllable = normalized_ipa_char
                        remaining_processed_ipa_chars = self.normalized_ipa_syllable(ipa_syllable[1:])
                        if remaining_processed_ipa_chars is not None:
                            normalized_ipa_syllable += remaining_processed_ipa_chars
                        else:
                            normalized_ipa_syllable = None

            else:
                # unmatched single character -- transform if possible
                if ipa_syllable in '_':
                    normalized_ipa_syllable = ipa_syllable
                elif ipa_syllable == 'R':
                    normalized_ipa_syllable = 'r'
                elif ipa_syllable == 'x':
                    normalized_ipa_syllable = 'ks'
                elif ipa_syllable == 'y':
                    normalized_ipa_syllable = 'U'
                elif ipa_syllable == 'c':
                    normalized_ipa_syllable = 'k'
                elif ipa_syllable == 'e':
                    normalized_ipa_syllable = 'E'
                elif ipa_syllable == 'a':
                    normalized_ipa_syllable = '&'
                elif ipa_syllable == 'W':
                    normalized_ipa_syllable = 'v'
                elif ipa_syllable == '(':
                    normalized_ipa_syllable = ''
                elif ipa_syllable == ')':
                    normalized_ipa_syllable = ''
                elif ipa_syllable == 'Y':
                    normalized_ipa_syllable = 'jU'
                elif ipa_syllable == 'V':
                    normalized_ipa_syllable = 'v'
                elif ipa_syllable == 'o':
                    normalized_ipa_syllable = 'O'
                elif ipa_syllable == '(':
                    normalized_ipa_syllable = ''
                elif ipa_syllable == ')':
                    normalized_ipa_syllable = ''
                else:
                    logging.warning("INTERNAL WARNING: No Deseret equivalent found for IPA syllable '%s'" % ipa_syllable)
                    normalized_ipa_syllable = None
        else:
            normalized_ipa_syllable = self.ipa_char_for_deseret_char(deseret_char)

        return normalized_ipa_syllable

    def get_ipa_word(self, deseret_word, dividers=False, english_phonetic=False):
        """Convert deseret word to IPA equivalent"""

        if not deseret_word:
            return None

        # remove apostrophes in contractions
        deseret_word = re.sub(r'[\',\\.]', '', deseret_word)

        # convert "o""e" to Oi
        oe_diphthong = chr(66564) + chr(66561)
        if oe_diphthong in deseret_word:
            deseret_word = deseret_word.replace(oe_diphthong, chr(66598))

        ipa_word = ''
        for deseret_letter in deseret_word:
            ipa_symbol = self.ipa_char_for_deseret_char(deseret_letter)
            if ipa_symbol is None:
                ipa_word = None
                break
            else:
                if english_phonetic:
                    ipa_symbol = self.ipa_to_phonetic.get(ipa_symbol, ipa_symbol)
                ipa_word += ('/' if dividers else '') + ipa_symbol

        if not ipa_word is None and dividers:
            ipa_word += '/'

        return ipa_word

    def lookup_ipa(self, ipa_word):
        english_word = self.pron_dict.get(ipa_word, None)
        return english_word

    def get_english_word(self, ipa_word):
        """Convert IPA word into English, if possible (trying alternate endings if no match is found)"""

        if not ipa_word:
            return None

        english_word = self.lookup_ipa(ipa_word)


        if not english_word and len(ipa_word) >= 2 and ipa_word[-2:] == "@z":
            english_word = self.lookup_ipa(ipa_word[:-2])
            if english_word:
                if len(english_word) >= 1 and english_word[:-1].lower() == "e":                    # e.g., "horses"
                    english_word += "s"
                else:                                                   # e.g., "bus"
                    english_word += "es"

        if not english_word and len(ipa_word) >= 1 and ipa_word[-1:] == "z":
            english_word = self.lookup_ipa(ipa_word[:-1])
            if english_word:
                if len(english_word) >= 1 and english_word[-1:] == "y":
                    english_word = english_word[:-1] + "ies"            # e.g., "candies"
                else:
                    english_word += "s"                                 # e.g., "funds"


        if not english_word and (ipa_word[-2:] == "iN" or ipa_word[-2:] == "IN"):
            english_word = self.lookup_ipa(ipa_word[:-2])
            if english_word:
                # word ends in "ing" sound
                # If it ends with <vowel>-<consonant>-e, drop the e and add -ing
                if len(english_word) >= 3 and english_word[-3].lower() in 'aeiou' and english_word[-2] not in 'aeiou' and english_word[-1] == 'e':
                    english_word = english_word[:-1] + "ing"            # e.g., "biking"
                # Else if it ends with <vowel>-<consonant>, repeat the last consonant and add -ing
                elif len(english_word) >= 2 and english_word[-2].lower() in 'aeiou' and english_word[-1] not in 'aeiou':
                    english_word += english_word[-1] + "ing"            # e.g., "digging"
                # Else add ing
                else:
                    english_word += "ing"                               # e.g., "seeing"

        if not english_word and len(ipa_word) >= 2 and ipa_word[-2:] == "li":
            english_word = self.lookup_ipa(ipa_word[:-2])
            if english_word:
                english_word += "ly"

        if not english_word and len(ipa_word) >= 2 and ipa_word[-2:] == "@d":
            english_word = self.lookup_ipa(ipa_word[:-2])
            if english_word:
                # word ends in "-ed"
                # If it ends with <consonant>-<vowel>-<consonant>, repeat last consonant and add -ed  (e.g., "fitted")
                if len(english_word) >= 3 and english_word[-3].lower() not in 'aeiou' and english_word[-2] in 'aeiou' and english_word[-1] not in 'aeiou':
                    english_word += english_word[-1] + "ed"
                # Else if it ends with e, add d
                elif len(english_word) >= 1 and english_word[-1].lower() == "e":                   # e.g., "stated"
                    english_word += "d"
                else:                                                   # e.g., "headed"
                    english_word += "ed"

        if not ipa_word and len(ipa_word) >= 1 and (ipa_word[-1:] == "d" or ipa_word[-1] == "t"):
            english_word = self.lookup_ipa(ipa_word[:-1])
            # If it ends in e, add d
            if english_word[-1] == "e":                                 # e.g., "raped"
                english_word += "d"
            # Else repeat the last consonant and add -ed
            else:
                english_word += english_word[-1] + "ed"                 # e.g., "capped"

        if not english_word and len(ipa_word) >= 2 and ipa_word[-2:] == "@r":
            english_word = self.lookup_ipa(ipa_word[:-2])
            if english_word:
                # word ends in "-er"
                # If it ends with <consonant>-<vowel>-<consonant>, repeat last consonant and add -er  (e.g., "runner")
                if len(english_word) >= 3 and english_word[-3].lower() not in 'aeiou' and english_word[-2] in 'aeiou' and english_word[-1] not in 'aeiou':
                    english_word += english_word[-1] + "er"
                # Else if it ends with e, add r
                elif len(english_word) >= 1 and english_word[-1].lower() == "e":                   # e.g., "hater"
                    english_word += "r"
                else:                                                   # e.g., "feeder"
                    english_word += "er"

        if not english_word and len(ipa_word) >= 1 and ipa_word[-1] == "s":
            english_word = self.lookup_ipa(ipa_word[:-1])
            if english_word:
                english_word += "s"

        return english_word


    def replace_ascii_chars_with_deseret_unicode(self, deseret_word):
        """
        AdamicBee font maps ASCII characters to Deseret characters;
        use the mapping to convert any ASCII characters to actual
        Deseret Unicode characters
        """

        if deseret_word is None:
            value = None
        else:
            value = ''
            for letter in deseret_word:
                value += self.ascii_to_deseret.get(letter, letter)

        return value
