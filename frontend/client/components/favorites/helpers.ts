import type { Favorite } from "@/lib/favorites-api";
import type { Agreement } from "@shared/agreement";

import type { FavoriteFilters } from "./types";

export function hasActiveFilters(f: FavoriteFilters): boolean {
  return (
    f.tagIds.length > 0 ||
    f.yearMin.trim() !== "" ||
    f.yearMax.trim() !== "" ||
    f.sizeMinUsd.trim() !== "" ||
    f.sizeMaxUsd.trim() !== "" ||
    f.target.trim() !== "" ||
    f.acquirer.trim() !== ""
  );
}

export function favoriteHref(fav: Favorite): string | null {
  const agreementUuid = fav.agreement_uuid;
  if (!agreementUuid) return null;
  if (fav.item_type === "section") {
    return `/agreements/${agreementUuid}?focusSectionUuid=${fav.item_uuid}`;
  }
  return `/agreements/${agreementUuid}`;
}

export function favoriteHeading(fav: Favorite, agreement: Agreement | null): string {
  const ctxTarget =
    agreement?.target ??
    (typeof fav.context?.target === "string"
      ? (fav.context.target as string)
      : null);
  const ctxAcquirer =
    agreement?.acquirer ??
    (typeof fav.context?.acquirer === "string"
      ? (fav.context.acquirer as string)
      : null);
  if (ctxTarget && ctxAcquirer) return `${ctxTarget} — ${ctxAcquirer}`;
  if (ctxTarget || ctxAcquirer) return ctxTarget ?? ctxAcquirer ?? "";
  return fav.item_uuid;
}

export function contextString(fav: Favorite, key: string): string | null {
  const value = fav.context?.[key];
  return typeof value === "string" && value.trim() ? value : null;
}

export function stripXmlText(xml: string | null | undefined): string {
  if (!xml) return "";
  if (typeof DOMParser !== "undefined") {
    const parsed = new DOMParser().parseFromString(xml, "text/xml");
    const parseError = parsed.querySelector("parsererror");
    if (!parseError) return parsed.documentElement.textContent ?? "";
  }
  return xml.replace(/<[^>]*>/g, " ");
}

export function firstWords(text: string, count: number): string {
  const words = text.replace(/\s+/g, " ").trim().split(" ").filter(Boolean);
  if (words.length <= count) return words.join(" ");
  return `${words.slice(0, count).join(" ")}…`;
}

export function formatDate(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString();
}
