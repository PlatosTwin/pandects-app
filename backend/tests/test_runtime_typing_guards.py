from pathlib import Path
import unittest


_RUNTIME_FILES = [
    "backend/app.py",
    "backend/models/auth.py",
    "backend/models/main_db.py",
    "backend/routes/auth.py",
    "backend/routes/agreements.py",
    "backend/routes/reference_data.py",
    "backend/routes/sections.py",
    "backend/services/sections_service.py",
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

    def test_route_deps_avoid_broad_callable_any_contracts(self) -> None:
        root = Path(__file__).resolve().parents[2]
        deps_path = root / "backend/routes/deps.py"
        allowed_callable_any_fields = {
            "_agreement_year_expr",
            "_coalesced_section_standard_ids",
        }

        violations: list[str] = []
        for line_no, line in enumerate(deps_path.read_text().splitlines(), start=1):
            if "Callable[..., Any]" not in line:
                continue
            if not any(field in line for field in allowed_callable_any_fields):
                violations.append(f"backend/routes/deps.py:{line_no}: {line.strip()}")

        self.assertFalse(
            violations,
            "Use explicit callable return types in deps contracts:\n" + "\n".join(violations),
        )

if __name__ == "__main__":
    unittest.main()
