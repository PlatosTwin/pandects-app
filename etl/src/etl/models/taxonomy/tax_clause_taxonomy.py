from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TaxClauseTaxonomyL2:
    label: str
    leaves: tuple[str, ...]


@dataclass(frozen=True)
class TaxClauseTaxonomyL1:
    label: str
    children: tuple[TaxClauseTaxonomyL2, ...]


TAX_CLAUSE_TAXONOMY: tuple[TaxClauseTaxonomyL1, ...] = (
    TaxClauseTaxonomyL1(
        label="Transaction Tax Status and Intended Treatment",
        children=(
            TaxClauseTaxonomyL2(
                label="Tax Classification of the Deal",
                leaves=(
                    "Intended Tax-Free Reorganization",
                    "Taxable Transaction / Stepped-Up Deal",
                    "Partnership / Disregarded Entity / Blocker Structure",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Tax Reporting Position",
                leaves=(
                    "Reporting Consistency with Intended Treatment",
                    "Allocation of Reporting Responsibility",
                    "No Contrary Filing or Position",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Tax Elections and Structural Actions",
                leaves=(
                    "338 / 336 Elections",
                    "Section 754 / Partnership Elections",
                    "Entity Classification Elections",
                    "Pre-Closing Restructuring / Conversion",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Tax Opinions and Tax Condition Support",
                leaves=(
                    "Tax Opinion Covenant",
                    "Cooperation for Tax Opinion / Tax Certificate",
                ),
            ),
        ),
    ),
    TaxClauseTaxonomyL1(
        label="Pre-Closing Tax Covenants",
        children=(
            TaxClauseTaxonomyL2(
                label="Ordinary-Course Tax Conduct",
                leaves=(
                    "Timely Filing of Tax Returns",
                    "Timely Payment of Taxes",
                    "Maintain Existing Tax Elections / Methods",
                    "No Unusual Tax Election or Accounting Method Change",
                    "No Amended Returns / Voluntary Disclosure / Ruling Request",
                    "No Waiver of Limitations / Extension of Assessment Period",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Restrictions on Tax Actions",
                leaves=(
                    "No 338 / Similar Election Without Consent",
                    "No Settlement of Tax Claims Without Consent",
                    "No Surrender of Refund Rights",
                    "No Change in Tax Residency / Classification",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Cooperation and Notice",
                leaves=(
                    "Notice of Tax Audits / Proceedings",
                    "Access to Tax Records and Workpapers",
                    "Interim Cooperation on Tax Matters",
                ),
            ),
        ),
    ),
    TaxClauseTaxonomyL1(
        label="Transfer, Withholding, and Transaction Taxes",
        children=(
            TaxClauseTaxonomyL2(
                label="Transfer Taxes",
                leaves=(
                    "Transfer / Sales / Use / Stamp Tax Allocation",
                    "Transfer Tax Return Filing Responsibility",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Withholding",
                leaves=(
                    "Purchase Price Withholding",
                    "Paying Agent / Exchange Agent Withholding",
                    "Dividend / Distribution Withholding",
                    "FIRPTA / Real Property Withholding",
                    "Backup Withholding / Forms Delivery",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Gross-Up and Tax Adjustment Mechanics",
                leaves=(
                    "Gross-Up Obligation",
                    "Tax Benefit / Tax Detriment Adjustment",
                    "Withholding Exception for Required Law",
                ),
            ),
        ),
    ),
    TaxClauseTaxonomyL1(
        label="Tax Returns, Contests, Refunds, and Post-Closing Administration",
        children=(
            TaxClauseTaxonomyL2(
                label="Tax Return Preparation",
                leaves=(
                    "Pre-Closing Return Responsibility",
                    "Straddle Return Preparation",
                    "Review / Comment / Consent Rights on Returns",
                    "Preparation Consistency / Past Practice",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Tax Contests",
                leaves=(
                    "Control of Pre-Closing Tax Contest",
                    "Control of Straddle / Shared Period Tax Contest",
                    "Notice, Participation, and Settlement Rights",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Refunds and Credits",
                leaves=(
                    "Ownership of Tax Refunds",
                    "Carryback / Carryforward Refund Rights",
                    "Refund Application / Payment Mechanics",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Cooperation and Records",
                leaves=(
                    "Record Retention",
                    "Mutual Cooperation After Closing",
                    "Access to Information for Tax Filings and Audits",
                ),
            ),
        ),
    ),
    TaxClauseTaxonomyL1(
        label="Tax Allocation, Indemnity, and Risk Shifting",
        children=(
            TaxClauseTaxonomyL2(
                label="Covered Taxes and Periods",
                leaves=(
                    "Pre-Closing Taxes",
                    "Straddle Period Tax Allocation",
                    "Taxes of Subsidiaries / Pre-Deal Entities",
                    "Successor / Transferee / Consolidated / Combined Group Liability",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Computation and Allocation Rules",
                leaves=(
                    "Closing Date Allocation Convention",
                    "Interim Closing of Books",
                    "Proration by Days / Other Allocation Formula",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Indemnity Mechanics",
                leaves=(
                    "Tax Indemnity / Reimbursement Obligation",
                    "Limitations / Survival / Exclusive Remedy Interaction",
                    "Treatment of Tax Benefits and Insurance Proceeds",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Payment and Procedural Rules",
                leaves=(
                    "Timing of Tax Indemnity Payments",
                    "Contest Control Under Tax Indemnity",
                    "Mitigation / Cooperation / Notice Conditions",
                ),
            ),
        ),
    ),
    TaxClauseTaxonomyL1(
        label="Tax Attributes, Compensation Tax, and Special Regimes",
        children=(
            TaxClauseTaxonomyL2(
                label="Tax Attributes",
                leaves=(
                    "Net Operating Losses / Credits / Basis / Earnings and Profits",
                    "Preservation or Use of Tax Attributes",
                    "Section 382 / 383 / Similar Limitation Matters",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Employment and Executive Compensation Tax",
                leaves=(
                    "280G / Golden Parachute",
                    "409A / Deferred Compensation",
                    "Payroll / Employment Tax Responsibility",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="International and Cross-Border Tax",
                leaves=(
                    "CFC / PFIC / Subpart F / GILTI-Related Matters",
                    "Tax Residency / Permanent Establishment",
                    "Treaty Claims / Cross-Border Withholding",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Special Regimes",
                leaves=(
                    "REIT / RIC / S Corporation / Partnership Status",
                    "Insurance / Energy / Industry-Specific Tax Regime",
                    "State and Local Tax Nexus / Apportionment",
                ),
            ),
        ),
    ),
    TaxClauseTaxonomyL1(
        label="Tax Representations",
        children=(
            TaxClauseTaxonomyL2(
                label="Returns and Payments",
                leaves=(
                    "Returns Filed and Taxes Paid",
                    "Adequate Accrual / Reserve for Taxes",
                    "Withholding Compliance",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Audits and Proceedings",
                leaves=(
                    "No Pending Audit / Examination / Claim",
                    "No Extension / Waiver of Limitations",
                    "No Deficiency / Lien / Closing Agreement / Ruling",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Attributes and Status",
                leaves=(
                    "Entity Classification / Residency / Special Status",
                    "Tax Sharing Agreement / Consolidated Group Membership",
                    "Availability of Tax Attributes",
                ),
            ),
            TaxClauseTaxonomyL2(
                label="Transaction-Specific Tax Reps",
                leaves=(
                    "No 280G Payments",
                    "FIRPTA Status",
                    "Reorganization / Intended Tax Treatment Support",
                ),
            ),
        ),
    ),
)


def iter_tax_clause_taxonomy_rows() -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str, str]],
    list[tuple[str, str, str]],
]:
    l1_rows: list[tuple[str, str]] = []
    l2_rows: list[tuple[str, str, str]] = []
    l3_rows: list[tuple[str, str, str]] = []

    for l1_index, l1 in enumerate(TAX_CLAUSE_TAXONOMY, start=1):
        l1_id = f"tax.{l1_index}"
        l1_rows.append((l1_id, l1.label))
        for l2_index, l2 in enumerate(l1.children, start=1):
            l2_id = f"{l1_id}.{l2_index}"
            l2_rows.append((l2_id, l2.label, l1_id))
            for l3_index, l3_label in enumerate(l2.leaves, start=1):
                l3_rows.append((f"{l2_id}.{l3_index}", l3_label, l2_id))

    return l1_rows, l2_rows, l3_rows


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _render_insert(table_name: str, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    values_sql = ",\n".join(
        "(" + ", ".join(_sql_string(value) for value in row) + ")" for row in rows
    )
    return (
        f"INSERT INTO {table_name} (" + ", ".join(columns) + ")\nVALUES\n" + values_sql + ";\n"
    )


def render_tax_clause_taxonomy_seed_sql(schema: str = "pdx") -> str:
    l1_rows, l2_rows, l3_rows = iter_tax_clause_taxonomy_rows()
    qualified_l1 = f"{schema}.tax_clause_taxonomy_l1"
    qualified_l2 = f"{schema}.tax_clause_taxonomy_l2"
    qualified_l3 = f"{schema}.tax_clause_taxonomy_l3"
    statements = [
        "-- Seed tax clause taxonomy reference data.",
        f"DELETE FROM {qualified_l3};",
        f"DELETE FROM {qualified_l2};",
        f"DELETE FROM {qualified_l1};",
        _render_insert(qualified_l1, ("standard_id", "label"), l1_rows).strip(),
        _render_insert(
            qualified_l2,
            ("standard_id", "label", "parent_id"),
            l2_rows,
        ).strip(),
        _render_insert(
            qualified_l3,
            ("standard_id", "label", "parent_id"),
            l3_rows,
        ).strip(),
    ]
    return "\n\n".join(statements) + "\n"


if __name__ == "__main__":
    print(render_tax_clause_taxonomy_seed_sql(), end="")
