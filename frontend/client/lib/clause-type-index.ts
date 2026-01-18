import type { ClauseTypeNode, ClauseTypeTree } from "@/lib/clause-types";

export function indexClauseTypePaths(
  tree: ClauseTypeTree,
): Record<string, readonly string[]> {
  const index: Record<string, readonly string[]> = {};

  const isClauseTypeNode = (
    value: ClauseTypeNode | ClauseTypeTree,
  ): value is ClauseTypeNode =>
    Object.prototype.hasOwnProperty.call(value, "id");

  const visit = (node: ClauseTypeTree, ancestors: string[]) => {
    for (const [label, value] of Object.entries(node)) {
      if (typeof value === "string") {
        index[value] = [...ancestors, label];
        continue;
      }
      if (isClauseTypeNode(value)) {
        index[value.id] = [...ancestors, label];
        if (value.children) {
          visit(value.children, [...ancestors, label]);
        }
      } else {
        visit(value, [...ancestors, label]);
      }
    }
  };

  visit(tree, []);
  return index;
}

export function indexClauseTypeLabels(
  tree: ClauseTypeTree,
): Record<string, string> {
  const index: Record<string, string> = {};

  const isClauseTypeNode = (
    value: ClauseTypeNode | ClauseTypeTree,
  ): value is ClauseTypeNode =>
    Object.prototype.hasOwnProperty.call(value, "id");

  const visit = (node: ClauseTypeTree) => {
    for (const [label, value] of Object.entries(node)) {
      if (typeof value === "string") {
        index[value] = label;
        continue;
      }
      if (isClauseTypeNode(value)) {
        index[value.id] = label;
        if (value.children) {
          visit(value.children);
        }
      } else {
        visit(value);
      }
    }
  };

  visit(tree);
  return index;
}
