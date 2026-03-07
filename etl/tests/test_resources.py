# pyright: reportAny=false
import unittest
from unittest.mock import patch

from etl.defs.resources import DBResource


class DBResourceTests(unittest.TestCase):
    def test_get_engine_enables_stale_connection_protection(self) -> None:
        db = DBResource(
            host="127.0.0.1",
            port="3306",
            user="user",
            password="password",
            database="pdx",
        )

        with patch("etl.defs.resources.create_engine") as mock_create_engine:
            _ = db.get_engine()

        mock_create_engine.assert_called_once_with(
            "mariadb+mysqldb://user:password@127.0.0.1:3306/pdx",
            pool_pre_ping=True,
            pool_recycle=3600,
        )


if __name__ == "__main__":
    _ = unittest.main()
