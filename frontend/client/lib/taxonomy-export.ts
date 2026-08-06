import type { TaxonomyLevel1 } from "@/lib/taxonomy-search";

const BRANCH = "├── ";
const LAST_BRANCH = "└── ";
const TRUNK = "│   ";
const GAP = "    ";

const line = (prefix: string, label: string, id: string, counts?: string) =>
  `${prefix}${label} [${id}]${counts ? ` (${counts})` : ""}`;

/**
 * Renders the taxonomy as an indented ASCII tree — the same L1 > L2 > L3
 * shape the tree view draws, in a form that survives a paste into a plain
 * text field. Counts mirror the labels shown on screen.
 */
export const formatTaxonomyTreeText = (entries: TaxonomyLevel1[]): string => {
  const lines: string[] = [];

  entries.forEach((entry) => {
    lines.push(
      line(
        "",
        entry.label,
        entry.id,
        `${entry.l2Count} groups, ${entry.l3Count} types`,
      ),
    );

    entry.children.forEach((child, childIndex) => {
      const isLastChild = childIndex === entry.children.length - 1;
      lines.push(
        line(
          isLastChild ? LAST_BRANCH : BRANCH,
          child.label,
          child.id,
          `${child.children.length} types`,
        ),
      );

      const leafTrunk = isLastChild ? GAP : TRUNK;
      child.children.forEach((leaf, leafIndex) => {
        const isLastLeaf = leafIndex === child.children.length - 1;
        lines.push(
          line(
            `${leafTrunk}${isLastLeaf ? LAST_BRANCH : BRANCH}`,
            leaf.label,
            leaf.id,
          ),
        );
      });
    });
  });

  return lines.join("\n");
};

/** e.g. `taxonomy_2026-08-06.txt` */
export const taxonomyTextFileName = (
  kind: "main" | "tax",
  today: Date = new Date(),
) => {
  const stamp = today.toISOString().split("T")[0];
  const base = kind === "tax" ? "tax_clause_taxonomy" : "taxonomy";
  return `${base}_${stamp}.txt`;
};
