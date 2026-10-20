import unittest

from backend.core.pagination import clamp_page_size, search_max_page_size


class ClampPageSizeTests(unittest.TestCase):
    def test_oversized_request_clamps_to_ceiling(self):
        self.assertEqual(clamp_page_size(10000, max_page_size=100), 100)
        self.assertEqual(clamp_page_size(10000, max_page_size=10), 10)

    def test_in_range_request_is_kept(self):
        self.assertEqual(clamp_page_size(40, max_page_size=100), 40)

    def test_non_positive_request_falls_back_to_default(self):
        self.assertEqual(clamp_page_size(0, max_page_size=100), 25)
        self.assertEqual(clamp_page_size(-3, max_page_size=10), 10)

    def test_tier_ceilings(self):
        self.assertEqual(search_max_page_size(is_authenticated=True), 100)
        self.assertEqual(search_max_page_size(is_authenticated=False), 10)


if __name__ == "__main__":
    unittest.main()
