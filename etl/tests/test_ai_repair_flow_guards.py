# pyright: reportAny=false
import unittest
from pathlib import Path

class AiRepairFlowGuardsTests(unittest.TestCase):
    def test_poll_asset_fails_fast_on_terminal_failed_batches(self) -> None:
        src = Path("src/etl/defs/d_ai_repair_asset.py").read_text(encoding="utf-8")
        self.assertIn("_AI_REPAIR_FAILED_BATCH_STATUSES", src)
        self.assertIn("terminal failed/cancelled/expired batches detected", src)
        self.assertIn("raise RuntimeError(", src)
        self.assertIn("return sorted(successful_request_ids)", src)

    def test_reconcile_is_scoped_to_polled_request_ids(self) -> None:
        src = Path("src/etl/defs/e_reconcile_tags.py").read_text(encoding="utf-8")
        self.assertIn("r.request_id IN :rids", src)
        self.assertIn("r.mode = 'full'", src)
        self.assertNotIn("label_error", src)
        self.assertNotIn("ai_repair_rulings_table", src)


if __name__ == "__main__":
    _ = unittest.main()
