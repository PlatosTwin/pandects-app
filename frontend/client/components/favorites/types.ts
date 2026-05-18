import type { FavoriteItemType } from "@/lib/favorites-api";

export type Filter = "all" | FavoriteItemType;

export const TYPE_LABELS: Record<FavoriteItemType, string> = {
  section: "Section",
  agreement: "Deal",
  tax_clause: "Tax clause",
};

export interface FavoriteFilters {
  tagIds: string[];
  yearMin: string;
  yearMax: string;
  sizeMinUsd: string;
  sizeMaxUsd: string;
  target: string;
  acquirer: string;
}

export interface SectionDetails {
  agreement_uuid: string | null;
  section_uuid: string;
  section_standard_id: string[];
  xml: string | null;
  article_title: string | null;
  section_title: string | null;
}

export const EMPTY_FILTERS: FavoriteFilters = {
  tagIds: [],
  yearMin: "",
  yearMax: "",
  sizeMinUsd: "",
  sizeMaxUsd: "",
  target: "",
  acquirer: "",
};

export const SIZE_OPTIONS = [
  { label: "$50M", value: "50000000" },
  { label: "$100M", value: "100000000" },
  { label: "$250M", value: "250000000" },
  { label: "$500M", value: "500000000" },
  { label: "$1B", value: "1000000000" },
  { label: "$2.5B", value: "2500000000" },
  { label: "$5B", value: "5000000000" },
  { label: "$10B", value: "10000000000" },
];

export const ANY_SIZE_VALUE = "any";
