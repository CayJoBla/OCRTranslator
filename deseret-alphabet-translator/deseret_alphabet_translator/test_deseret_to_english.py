# -*- coding: utf-8 -*-
import unittest
from deseret_to_english import DeseretToEnglish
from english_to_deseret import EnglishToDeseret

class TestDeseretToEnglish(unittest.TestCase):

    def setUp(self):
        self.deseret = EnglishToDeseret()
        self.english = DeseretToEnglish()


    def english_to_deseret(self, english):
        deseret_html = self.deseret.translate(english)
        unicode_deseret = DeseretToEnglish.unescape(deseret_html)
        return unicode_deseret


    def test_translate(self):

        words = [#'hello',
                 'world', 'hello world', 'hello, world']

        for word in words:
            deseret_word = self.english_to_deseret(word)
            english_word = self.english.translate(deseret_word)
            self.assertEqual(english_word, word)


    def test_translate_word(self):

        words = ['jellies', 'horse',
                 'horses',
                 #'buys',       # buys translates as 'byes' -- not much to be done without analyzing semantics
                 'candies', 'buses', 'biking', 'digging', 'runs', 'seeing', 'Deseret']

        for word in words:
            deseret_word = self.english_to_deseret(word)
            english_word = self.english.translate_word(deseret_word)
            self.assertEqual(english_word, word)

    def test_get_ipa_word(self):
        ipa_word = self.english.get_ipa_word(self.english_to_deseret('hello'))
        self.assertEqual(ipa_word, "hEloU")

    def test_get_english_word(self):

        ipa_word = self.english.get_ipa_word(self.english_to_deseret('hello'))
        english_word = self.english.get_english_word(ipa_word)
        self.assertEqual(english_word, 'hello')

    def test_check_plural(self):
        self.failUnless(True)

if __name__ == '__main__':
    unittest.main()
