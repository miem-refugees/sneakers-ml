from unittest import TestCase

from sneakers_ml.bot.utils.escape import escape


class TestEscape(TestCase):
    def test_simple_escapes(self):
        assert escape("test") == "test"
        assert escape("") == ""
        assert escape(r"_*[]()~>#\+\-=|{}.!") == "\\_\\*\\[\\]\\(\\)\\~\\>\\#\\\\+\\\\-\\=\\|\\{\\}\\.\\!"
