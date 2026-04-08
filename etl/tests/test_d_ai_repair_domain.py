# pyright: reportAny=false, reportPrivateUsage=false
import unittest

from etl.domain.d_ai_repair import (
    RepairDecision,
    _system_prompt_full,
    _user_prompt_full,
    build_jsonl_lines_for_page,
)


class AiRepairDomainTests(unittest.TestCase):
    def test_full_mode_request_id_includes_xml_version(self) -> None:
        decision = RepairDecision(mode="full", windows=[(0, 12)], token_map=[])
        lines, metas = build_jsonl_lines_for_page(
            page_uuid="11111111-1111-1111-1111-111111111111",
            text="Sample text.",
            decision=decision,
            model="gpt-5.1",
            uncertain_spans=[],
            xml_version=7,
        )

        self.assertEqual(len(lines), 1)
        self.assertEqual(len(metas), 1)
        self.assertEqual(
            lines[0]["custom_id"],
            "11111111-1111-1111-1111-111111111111::full::7",
        )
        self.assertEqual(
            metas[0]["request_id"],
            "11111111-1111-1111-1111-111111111111::full::7",
        )

    def test_full_mode_user_prompt_is_raw_page_text_only(self) -> None:
        page_text = "Line 1\nLine 2"
        prompt = _user_prompt_full(
            page_uuid="11111111-1111-1111-1111-111111111111",
            text=page_text,
        )
        self.assertEqual(prompt, page_text)
        self.assertNotIn("PAGE_UUID=", prompt)
        self.assertNotIn("Task: Insert", prompt)

    def test_full_mode_user_prompt_can_include_toc_context(self) -> None:
        prompt = _user_prompt_full(
            page_uuid="11111111-1111-1111-1111-111111111111",
            text="5.22 Customers and Suppliers.",
            toc_context="Article 5 TOC section numbering: 5.20, 5.21, 5.22, 5.23",
        )

        self.assertEqual(prompt, "5.22 Customers and Suppliers.")

    def test_full_mode_request_body_omits_toc_context_by_default(self) -> None:
        decision = RepairDecision(mode="full", windows=[(0, 12)], token_map=[])
        lines, _ = build_jsonl_lines_for_page(
            page_uuid="11111111-1111-1111-1111-111111111111",
            text="Sample text.",
            decision=decision,
            model="gpt-5.1",
            uncertain_spans=[],
            xml_version=7,
        )

        prompt = lines[0]["body"]["input"][0]["content"]
        self.assertEqual(prompt, "Sample text.")

    def test_full_mode_request_body_includes_toc_context_when_provided(self) -> None:
        decision = RepairDecision(mode="full", windows=[(0, 28)], token_map=[])
        lines, _ = build_jsonl_lines_for_page(
            page_uuid="11111111-1111-1111-1111-111111111111",
            text="5.22 Customers and Suppliers.",
            decision=decision,
            model="gpt-5.1",
            uncertain_spans=[],
            xml_version=7,
            toc_context="Article 5 TOC section numbering: 5.20, 5.21, 5.22, 5.23",
        )

        prompt = lines[0]["body"]["input"][0]["content"]
        instructions = lines[0]["body"]["instructions"]
        self.assertEqual(prompt, "5.22 Customers and Suppliers.")
        self.assertIn("Article 5 TOC section numbering: 5.20, 5.21, 5.22, 5.23", instructions)
        self.assertIn("All offsets and selected_text values must still be based only on the provided page text.", instructions)
        self.assertIn("If the exact section number appears on this page as a standalone block heading", instructions)
        self.assertIn("Pay especially close attention to text near the top of the page", instructions)
        self.assertIn("appears alone on its own line or clearly starts a new block", instructions)
        self.assertIn("Do not tag inline prose references", instructions)

    def test_full_mode_system_prompt_biases_toward_precision(self) -> None:
        prompt = _system_prompt_full()
        self.assertIn("False positives are worse than missed spans.", prompt)
        self.assertIn("Do not infer, reconstruct, or borrow headings", prompt)
        self.assertIn("Bare numeric headings like", prompt)
        self.assertIn("it is probably a reference rather than a heading", prompt)


if __name__ == "__main__":
    _ = unittest.main()
