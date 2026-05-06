import { apiUrl } from "@/lib/api-config";
import { authFetchJson } from "@/lib/auth-fetch";

export type FavoriteItemType = "section" | "agreement" | "tax_clause";

export const TAG_COLORS = [
  "slate",
  "red",
  "orange",
  "amber",
  "green",
  "teal",
  "blue",
  "violet",
] as const;
export type TagColor = (typeof TAG_COLORS)[number];

export interface FavoriteTag {
  id: string;
  name: string;
  color: TagColor;
  created_at?: string | null;
}

export interface Favorite {
  id: string;
  project_id: string;
  item_type: FavoriteItemType;
  item_uuid: string;
  agreement_uuid: string | null;
  note: string | null;
  context: Record<string, unknown> | null;
  tags: FavoriteTag[];
  created_at: string | null;
  updated_at: string | null;
}

export interface FavoriteProject {
  id: string;
  name: string;
  is_default: boolean;
  sort_order: number;
  created_at: string | null;
}

export interface FavoritesListResponse {
  favorites: Favorite[];
}

export interface FavoritesExistsResponse {
  favorites: Record<string, string>;
}

export interface FavoriteProjectsResponse {
  projects: FavoriteProject[];
}

export interface FavoriteCreateInput {
  item_type: FavoriteItemType;
  item_uuid: string;
  project_id?: string | null;
  note?: string | null;
  context?: Record<string, unknown> | null;
}

export interface FavoriteUpsertResponse {
  favorite: Favorite;
  created: boolean;
}

export async function listFavorites(): Promise<Favorite[]> {
  const res = await authFetchJson<FavoritesListResponse>(apiUrl("v1/me/favorites"));
  return res.favorites;
}

export async function listFavoriteProjects(): Promise<FavoriteProject[]> {
  const res = await authFetchJson<FavoriteProjectsResponse>(
    apiUrl("v1/me/favorite-projects"),
  );
  return res.projects;
}

export async function favoritesExists(
  itemType: FavoriteItemType,
  itemUuids: string[],
): Promise<Record<string, string>> {
  if (itemUuids.length === 0) return {};
  const params = new URLSearchParams({
    item_type: itemType,
    item_uuids: itemUuids.join(","),
  });
  const res = await authFetchJson<FavoritesExistsResponse>(
    apiUrl(`v1/me/favorites/exists?${params.toString()}`),
  );
  return res.favorites;
}

export async function upsertFavorite(
  input: FavoriteCreateInput,
): Promise<FavoriteUpsertResponse> {
  const response = await authFetchJson<FavoriteUpsertResponse>(
    apiUrl("v1/me/favorites?view=minimal"),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    },
  );
  return {
    ...response,
    favorite: {
      ...response.favorite,
      agreement_uuid: null,
      tags: [],
    },
  };
}

export async function patchFavorite(
  id: string,
  input: { note?: string | null; project_id?: string | null },
): Promise<{ favorite: Favorite }> {
  return authFetchJson<{ favorite: Favorite }>(apiUrl(`v1/me/favorites/${id}`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

export async function deleteFavorite(id: string): Promise<void> {
  await authFetchJson<{ deleted: boolean }>(apiUrl(`v1/me/favorites/${id}`), {
    method: "DELETE",
  });
}

export async function listTags(): Promise<FavoriteTag[]> {
  const res = await authFetchJson<{ tags: FavoriteTag[] }>(apiUrl("v1/me/tags"));
  return res.tags;
}

export async function createTag(input: {
  name: string;
  color?: TagColor;
}): Promise<{ tag: FavoriteTag; created: boolean }> {
  return authFetchJson<{ tag: FavoriteTag; created: boolean }>(
    apiUrl("v1/me/tags"),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    },
  );
}

export async function patchTag(
  id: string,
  input: { name?: string; color?: TagColor },
): Promise<FavoriteTag> {
  const res = await authFetchJson<{ tag: FavoriteTag }>(
    apiUrl(`v1/me/tags/${id}`),
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    },
  );
  return res.tag;
}

export async function deleteTag(id: string): Promise<void> {
  await authFetchJson<{ deleted: boolean }>(apiUrl(`v1/me/tags/${id}`), {
    method: "DELETE",
  });
}

export async function setFavoriteTags(
  favoriteId: string,
  tagIds: string[],
): Promise<FavoriteTag[]> {
  const res = await authFetchJson<{ tags: FavoriteTag[] }>(
    apiUrl(`v1/me/favorites/${favoriteId}/tags`),
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tag_ids: tagIds }),
    },
  );
  return res.tags;
}
