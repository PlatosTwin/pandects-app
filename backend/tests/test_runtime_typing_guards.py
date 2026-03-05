from pathlib import Path
import unittest


_RUNTIME_FILES = [
    "backend/app.py",
    "backend/models/auth.py",
    "backend/models/main_db.py",
    "backend/routes/auth.py",
    "backend/routes/agreements.py",
    "backend/routes/reference_data.py",
    "backend/routes/search.py",
    "backend/services/search_service.py",
    "backend/services/usage.py",
]


class RuntimeTypingGuardTests(unittest.TestCase):
    def test_runtime_files_have_no_broad_pyright_file_suppression(self) -> None:
        failures: list[str] = []
        root = Path(__file__).resolve().parents[2]
        for rel_path in _RUNTIME_FILES:
            path = root / rel_path
            if not path.exists():
                continue
            for line_no, line in enumerate(path.read_text().splitlines(), start=1):
                stripped = line.strip()
                if not stripped.startswith("# pyright:"):
                    continue
                failures.append(f"{rel_path}:{line_no}: {stripped}")
        self.assertFalse(
            failures,
            "Remove broad file-level pyright suppressions:\n" + "\n".join(failures),
        )


if __name__ == "__main__":
    unittest.main()
