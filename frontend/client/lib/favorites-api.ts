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
  project_ids: string[];
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
  color: TagColor;
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

export async function listFavoritesForProject(
  projectId: string,
): Promise<Favorite[]> {
  const params = new URLSearchParams({ project_id: projectId });
  const res = await authFetchJson<FavoritesListResponse>(
    apiUrl(`v1/me/favorites?${params.toString()}`),
  );
  return res.favorites;
}

export async function listFavoriteProjects(): Promise<FavoriteProject[]> {
  const res = await authFetchJson<FavoriteProjectsResponse>(
    apiUrl("v1/me/favorite-projects"),
  );
  return res.projects;
}

export async function createProject(input: {
  name: string;
  color?: TagColor;
}): Promise<FavoriteProject> {
  const res = await authFetchJson<{
    project: FavoriteProject;
    created: boolean;
  }>(apiUrl("v1/me/favorite-projects"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return res.project;
}

export async function patchProject(
  id: string,
  input: { name?: string; color?: TagColor; sort_order?: number },
): Promise<FavoriteProject> {
  const res = await authFetchJson<{ project: FavoriteProject }>(
    apiUrl(`v1/me/favorite-projects/${id}`),
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    },
  );
  return res.project;
}

export async function deleteProject(
  id: string,
  reassignProjectId?: string,
): Promise<{
  deleted: boolean;
  reassigned_to_project_id: string;
  moved: number;
}> {
  const params = reassignProjectId
    ? `?${new URLSearchParams({ reassign_project_id: reassignProjectId }).toString()}`
    : "";
  return authFetchJson<{
    deleted: boolean;
    reassigned_to_project_id: string;
    moved: number;
  }>(apiUrl(`v1/me/favorite-projects/${id}${params}`), {
    method: "DELETE",
  });
}

export async function bulkMoveFavorites(
  favoriteIds: string[],
  projectId: string,
): Promise<{ project_id: string; favorite_ids: string[]; moved: number }> {
  return authFetchJson<{
    project_id: string;
    favorite_ids: string[];
    moved: number;
  }>(apiUrl("v1/me/favorites/bulk-move"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ favorite_ids: favoriteIds, project_id: projectId }),
  });
}

export async function bulkCopyFavorites(
  favoriteIds: string[],
  projectIds: string[],
): Promise<{ project_ids: string[]; favorite_ids: string[]; copied: number }> {
  return authFetchJson<{
    project_ids: string[];
    favorite_ids: string[];
    copied: number;
  }>(apiUrl("v1/me/favorites/bulk-copy"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ favorite_ids: favoriteIds, project_ids: projectIds }),
  });
}

export async function bulkUpdateFavoriteTags(
  favoriteIds: string[],
  tagIds: string[],
  action: "add" | "remove",
): Promise<{
  action: "add" | "remove";
  tag_ids: string[];
  favorite_ids: string[];
  tags_by_favorite: Record<string, FavoriteTag[]>;
  updated: number;
}> {
  return authFetchJson<{
    action: "add" | "remove";
    tag_ids: string[];
    favorite_ids: string[];
    tags_by_favorite: Record<string, FavoriteTag[]>;
    updated: number;
  }>(apiUrl("v1/me/favorites/bulk-tags"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      favorite_ids: favoriteIds,
      tag_ids: tagIds,
      action,
    }),
  });
}

export async function setFavoriteProjects(
  favoriteId: string,
  projectIds: string[],
): Promise<string[]> {
  const res = await authFetchJson<{ project_ids: string[] }>(
    apiUrl(`v1/me/favorites/${favoriteId}/projects`),
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_ids: projectIds }),
    },
  );
  return res.project_ids;
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
      project_ids: response.favorite.project_ids ?? [response.favorite.project_id],
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

export async function getFavoriteTags(favoriteId: string): Promise<FavoriteTag[]> {
  const res = await authFetchJson<{ tags: FavoriteTag[] }>(
    apiUrl(`v1/me/favorites/${favoriteId}/tags`),
  );
  return res.tags;
}
