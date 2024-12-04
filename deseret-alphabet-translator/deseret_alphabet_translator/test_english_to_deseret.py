import unittest
from english_to_deseret import EnglishToDeseret

class TestEnglishToDeseret(unittest.TestCase):

    def setUp(self):
        self.deseret = EnglishToDeseret()

    def test_translate(self):
        deseret_word = self.deseret.translate('hello')
        self.assertEqual(deseret_word, "𐐸𐐯𐑊𐐬")

        deseret_word = self.deseret.translate('world')
        self.assertEqual(deseret_word, "𐐶𐐲𐑉𐑊𐐼")

        deseret_word = self.deseret.translate('hello world')
        self.assertEqual(deseret_word, "𐐸𐐯𐑊𐐬 𐐶𐐲𐑉𐑊𐐼")

        deseret_word = self.deseret.translate('hello, world')
        self.assertEqual(deseret_word, "𐐸𐐯𐑊𐐬, 𐐶𐐲𐑉𐑊𐐼")

    def test_translate_word(self):
        deseret_word = self.deseret.translate('jellies')
        self.assertEqual(deseret_word, '𐐾𐐯𐑊𐐨𐑆')

        deseret_word = self.deseret.translate('horse')
        self.assertEqual(deseret_word, '𐐸𐐫𐑉𐑅')

        deseret_word = self.deseret.translate('horses')
        self.assertEqual(deseret_word, '𐐸𐐫𐑉𐑅𐑆')

        deseret_word = self.deseret.translate('buys')
        self.assertEqual(deseret_word, '𐐺𐐴𐑆')

        deseret_word = self.deseret.translate('candies')
        self.assertEqual(deseret_word, '𐐿𐐰𐑌𐐼𐐨𐑆')

        deseret_word = self.deseret.translate('buses')
        self.assertEqual(deseret_word, '𐐺𐐲𐑅𐐲𐑆')

        deseret_word = self.deseret.translate('biking')
        self.assertEqual(deseret_word, '𐐺𐐴𐐿𐐮𐑍')

        deseret_word = self.deseret.translate('digging')
        self.assertEqual(deseret_word, '𐐼𐐮𐑀𐐮𐑍')

        deseret_word = self.deseret.translate('runs')
        self.assertEqual(deseret_word, '𐑉𐐲𐑌𐑆')

        deseret_word = self.deseret.translate('seeing')
        self.assertEqual(deseret_word, '𐑅𐐨𐐨𐑍')

        deseret_word = self.deseret.translate('Deseret')
        self.assertEqual(deseret_word, '𐐔𐐯𐑅𐐨𐑉𐐯𐐻')

    def test_get_ipa_word(self):
        ipa_word = self.deseret.get_ipa_word('hello')
        self.assertEqual(ipa_word, "h/E/'l/oU/")

    def test_get_deseret_word(self):
        ipa_word = self.deseret.get_ipa_word('hello')
        deseret_word = self.deseret.get_deseret_word(ipa_word)
        self.assertEqual(deseret_word,  u'\U00010410\U00010407\U00010422\U00010404')

    def test_translate_capitalization(self):
        deseret_word = self.deseret.translate('Hello')
        self.assertEqual(deseret_word, "𐐐𐐯𐑊𐐬")

        deseret_word = self.deseret.translate('woRlD')
        self.assertEqual(deseret_word, "𐐶𐐲𐑉𐑊𐐼")

        deseret_word = self.deseret.translate('HELLO world')
        self.assertEqual(deseret_word, "𐐐𐐇𐐢𐐄 𐐶𐐲𐑉𐑊𐐼")

        deseret_word = self.deseret.translate('Hello, WORLD')
        self.assertEqual(deseret_word, "𐐐𐐯𐑊𐐬, 𐐎𐐊𐐡𐐢𐐔")


if __name__ == '__main__':
    unittest.main()