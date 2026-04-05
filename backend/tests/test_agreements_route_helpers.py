import unittest
from decimal import Decimal

from backend.routes.agreements import _to_float_or_none


class AgreementsRouteHelperTests(unittest.TestCase):
    def test_to_float_or_none_accepts_decimal(self) -> None:
        self.assertEqual(_to_float_or_none(Decimal("90.3")), 90.3)


if __name__ == "__main__":
    _ = unittest.main()
