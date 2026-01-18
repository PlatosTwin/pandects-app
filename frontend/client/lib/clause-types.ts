export interface ClauseTypeNode {
  id: string;
  children?: ClauseTypeTree;
}

export interface ClauseTypeTree {
  [key: string]: string | ClauseTypeNode | ClauseTypeTree;
}
