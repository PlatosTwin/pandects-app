// Agreement Types

export interface Agreement {
  year: string;
  target: string;
  acquirer: string;
  xml: string;
}

// Table of Contents types for XML navigation
export interface TOCItem {
  id: string;
  title: string;
  level: number;
  sectionUuid?: string;
  children?: TOCItem[];
}
