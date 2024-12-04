__author__ = 'nathan'

import unittest
import deseret_strings


class MyTestCase(unittest.TestCase):
    def test_upper(self):

        # built-in Python unicode operations
        small_long_i = chr(66600)      # small long i
        upper_long_i = chr(66560)      # capital long i
        self.assertEqual(small_long_i.upper(), upper_long_i)
        self.assertEqual(False, small_long_i.isupper())
        self.assertEqual(True, upper_long_i.isupper())


        leading_upper = upper_long_i + small_long_i
        all_upper = upper_long_i + upper_long_i
        all_lower = small_long_i + small_long_i
        trailing_upper = small_long_i + upper_long_i

        multi_leading_upper = upper_long_i + upper_long_i + small_long_i

        # is_leading_upper
        self.assertEqual(True,  deseret_strings.is_leading_upper(leading_upper))
        self.assertEqual(False, deseret_strings.is_leading_upper(all_upper))
        self.assertEqual(False, deseret_strings.is_leading_upper(all_lower))
        self.assertEqual(False, deseret_strings.is_leading_upper(trailing_upper))
        self.assertEqual(True,  deseret_strings.is_leading_upper(multi_leading_upper))
        self.assertEqual(True,  deseret_strings.is_leading_upper(multi_leading_upper + upper_long_i))

        # is_all_upper
        self.assertEqual(False, deseret_strings.is_all_upper(leading_upper))
        self.assertEqual(True,  deseret_strings.is_all_upper(all_upper))
        self.assertEqual(False, deseret_strings.is_all_upper(all_lower))
        self.assertEqual(False, deseret_strings.is_all_upper(trailing_upper))


if __name__ == '__main__':
    unittest.main()
