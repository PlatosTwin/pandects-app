from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from etl.utils.db_env import build_engine_from_env


class DbEnvTests(unittest.TestCase):
    @patch("etl.utils.db_env.create_engine")
    def test_build_engine_defaults_mariadb_port_to_3306(
        self,
        create_engine: MagicMock,
    ) -> None:
        with patch.dict(
            os.environ,
            {
                "MARIADB_USER": "user",
                "MARIADB_PASSWORD": "password",
                "MARIADB_HOST": "127.0.0.1",
                "MARIADB_DATABASE": "pdx",
            },
            clear=True,
        ):
            _ = build_engine_from_env()

        create_engine.assert_called_once_with(
            "mariadb+mysqldb://user:password@127.0.0.1:3306/pdx"
        )


if __name__ == "__main__":
    _ = unittest.main()
