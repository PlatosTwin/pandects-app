# pyright: reportAny=false
import unittest

from etl.utils.batch_keys import agreement_batch_key, agreement_version_batch_key


class BatchKeysTests(unittest.TestCase):
    def test_agreement_batch_key_is_order_insensitive(self) -> None:
        key_a = agreement_batch_key(["b", "a", "a"])
        key_b = agreement_batch_key(["a", "b"])
        self.assertEqual(key_a, key_b)

    def test_agreement_version_batch_key_is_order_insensitive(self) -> None:
        key_a = agreement_version_batch_key([("a", 2), ("a", 1), ("b", 1), ("a", 2)])
        key_b = agreement_version_batch_key([("b", 1), ("a", 1), ("a", 2)])
        self.assertEqual(key_a, key_b)

    def test_agreement_version_batch_key_is_version_sensitive(self) -> None:
        key_a = agreement_version_batch_key([("a", 1)])
        key_b = agreement_version_batch_key([("a", 2)])
        self.assertNotEqual(key_a, key_b)


if __name__ == "__main__":
    _ = unittest.main()
