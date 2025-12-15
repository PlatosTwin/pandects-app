import type { ClauseTypeTree } from "@/lib/clause-types";

export function indexClauseTypePaths(
  tree: ClauseTypeTree,
): Record<string, readonly string[]> {
  const index: Record<string, readonly string[]> = {};

  const visit = (node: ClauseTypeTree, ancestors: string[]) => {
    for (const [label, value] of Object.entries(node)) {
      if (typeof value === "string") {
        index[value] = [...ancestors, label];
        continue;
      }
      visit(value, [...ancestors, label]);
    }
  };

  visit(tree, []);
  return index;
}

