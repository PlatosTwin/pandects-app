import unittest

from etl.domain.xml_tag_repairs import move_split_word_tails_into_headings


def _repair(text: str) -> str:
    return move_split_word_tails_into_headings(text)[0]


class SplitWordTailRepairTests(unittest.TestCase):
    def test_moves_letter_tail_after_digit(self) -> None:
        self.assertEqual(
            _repair("<section>Section 5.05 Section 280</section>G As soon as practicable"),
            "<section>Section 5.05 Section 280G</section> As soon as practicable",
        )
        self.assertEqual(
            _repair("<section>6.7 Section 280</section>G\n\n.\n\nThe Company"),
            "<section>6.7 Section 280G</section>\n\n.\n\nThe Company",
        )
        self.assertEqual(
            _repair("<section>Section 5.05 Section 280</section>G"),
            "<section>Section 5.05 Section 280G</section>",
        )

    def test_moves_digit_led_tail_after_letter(self) -> None:
        self.assertEqual(
            _repair("<article>ARTICLE III REPRESENTATIONS AND WARRANTIES OF H</article>2EG\n\n3.1"),
            "<article>ARTICLE III REPRESENTATIONS AND WARRANTIES OF H2EG</article>\n\n3.1",
        )

    def test_reports_stats(self) -> None:
        _, stats = move_split_word_tails_into_headings("<section>5.8 Section 280</section>G Text")
        self.assertEqual(stats.applied["move_split_word_tails_into_headings"], 1)

    def test_leaves_sentence_start_glue(self) -> None:
        cases = [
            # A sentence glued to an untitled heading starts with a word, not a tail.
            "<section>Section 5</section>In the event",
            "<section>Section 3.1</section>A Party may",
            "<section>Section 3.1</section>The Company shall",
            "<section>Section 9.1</section>IN WITNESS WHEREOF",
            "<section>Section 9.1</section>THE PARTIES",
            "<section>Section 9.1</section>NOTICES. All notices",
            "<section>Section 2.19 Insurance Policies</section>A description",
            "<article>13 Assignment and transfer</article>This Agreement",
            # The tag already sits on a space or punctuation boundary.
            "<section>Section 280</section> G",
            "<section>Section 4.1.</section>G Something",
            # The tail continues with punctuation, so it is not a whole glued word.
            "<section>Section 280</section>G. Text",
            # A bare number after a letter is a separate token.
            "<article>ARTICLE</article>5 Covenants",
            "<section>Section 4.1 Taxes</section>2018 returns",
            # Every observed tail is one letter; an acronym after the tag is a word.
            "<section>Section 6.3</section>ABC shall",
            "<section>Section 1.2</section>US GAAP",
            "<section>Section 4</section>III",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(_repair(text), text)

    def test_leaves_text_without_heading_tags(self) -> None:
        text = "Section 280G of the Code and Section 409A apply. H2EG"
        self.assertEqual(_repair(text), text)


if __name__ == "__main__":
    unittest.main()
