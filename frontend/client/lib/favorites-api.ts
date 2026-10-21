import type { components } from "@shared/api.gen";
import { apiUrl } from "@/lib/api-config";
import { apiFetchJson } from "@/lib/api/client";

// Response types are generated from the backend spec (shared/api.gen.ts).
type Schemas = components["schemas"];

export type FavoriteItemType = Schemas["FavoriteCreate"]["item_type"];

export const TAG_COLORS = [
  "slate",
  "red",
  "orange",
  "amber",
  "green",
  "teal",
  "blue",
  "violet",
] as const satisfies readonly Schemas["FavoriteTag"]["color"][];
export type TagColor = (typeof TAG_COLORS)[number];

/** A tag as attached to a favorite; `created_at` only comes with `GET /v1/me/tags`. */
export type FavoriteTag = Schemas["FavoriteTagRef"] &
  Partial<Pick<Schemas["FavoriteTag"], "created_at">>;

export type Favorite = Schemas["Favorite"];

export type FavoriteProject = Schemas["FavoriteProject"];

export type FavoritesListResponse = Schemas["FavoritesListResponse"];

export type FavoritesExistsResponse = Schemas["FavoritesExistsResponse"];

export type FavoriteProjectsResponse = Schemas["FavoriteProjectsResponse"];

export type FavoriteCreateInput = Schemas["FavoriteCreate"];

/** `FavoriteCreateResponse` with the minimal favorite filled out to a full `Favorite`. */
export type FavoriteUpsertResponse = Omit<Schemas["FavoriteCreateResponse"], "favorite"> & {
  favorite: Favorite;
};

export async function listFavorites(): Promise<Favorite[]> {
  const res = await apiFetchJson<FavoritesListResponse>(apiUrl("v1/me/favorites"));
  return res.favorites;
}

export async function listFavoritesForProject(
  projectId: string,
): Promise<Favorite[]> {
  const params = new URLSearchParams({ project_id: projectId });
  const res = await apiFetchJson<FavoritesListResponse>(
    apiUrl(`v1/me/favorites?${params.toString()}`),
  );
  return res.favorites;
}

export async function listFavoriteProjects(): Promise<FavoriteProject[]> {
  const res = await apiFetchJson<FavoriteProjectsResponse>(
    apiUrl("v1/me/favorite-projects"),
  );
  return res.projects;
}

export async function createProject(
  input: Schemas["ProjectCreate"],
): Promise<FavoriteProject> {
  const res = await apiFetchJson<Schemas["FavoriteProjectCreateResponse"]>(apiUrl("v1/me/favorite-projects"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return res.project;
}

export async function patchProject(
  id: string,
  input: Schemas["ProjectUpdate"],
): Promise<FavoriteProject> {
  const res = await apiFetchJson<Schemas["FavoriteProjectResponse"]>(
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
): Promise<Schemas["FavoriteProjectDeleteResponse"]> {
  const params = reassignProjectId
    ? `?${new URLSearchParams({ reassign_project_id: reassignProjectId }).toString()}`
    : "";
  return apiFetchJson<Schemas["FavoriteProjectDeleteResponse"]>(apiUrl(`v1/me/favorite-projects/${id}${params}`), {
    method: "DELETE",
  });
}

export async function bulkMoveFavorites(
  favoriteIds: string[],
  projectId: string,
): Promise<Schemas["FavoritesBulkMoveResponse"]> {
  return apiFetchJson<Schemas["FavoritesBulkMoveResponse"]>(apiUrl("v1/me/favorites/bulk-move"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ favorite_ids: favoriteIds, project_id: projectId }),
  });
}

export async function bulkCopyFavorites(
  favoriteIds: string[],
  projectIds: string[],
): Promise<Schemas["FavoritesBulkCopyResponse"]> {
  return apiFetchJson<Schemas["FavoritesBulkCopyResponse"]>(apiUrl("v1/me/favorites/bulk-copy"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ favorite_ids: favoriteIds, project_ids: projectIds }),
  });
}

export async function bulkUpdateFavoriteTags(
  favoriteIds: string[],
  tagIds: string[],
  action: Schemas["FavoritesBulkTags"]["action"],
): Promise<Schemas["FavoritesBulkTagsResponse"]> {
  return apiFetchJson<Schemas["FavoritesBulkTagsResponse"]>(apiUrl("v1/me/favorites/bulk-tags"), {
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
  const res = await apiFetchJson<Schemas["FavoriteProjectIdsResponse"]>(
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
  const res = await apiFetchJson<FavoritesExistsResponse>(
    apiUrl(`v1/me/favorites/exists?${params.toString()}`),
  );
  return res.favorites;
}

export async function upsertFavorite(
  input: FavoriteCreateInput,
): Promise<FavoriteUpsertResponse> {
  const response = await apiFetchJson<Schemas["FavoriteCreateResponse"]>(
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
  input: Schemas["FavoriteUpdate"],
): Promise<Schemas["FavoriteResponse"]> {
  return apiFetchJson<Schemas["FavoriteResponse"]>(apiUrl(`v1/me/favorites/${id}`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

export async function deleteFavorite(id: string): Promise<void> {
  await apiFetchJson<Schemas["FavoriteDeletedResponse"]>(apiUrl(`v1/me/favorites/${id}`), {
    method: "DELETE",
  });
}

export async function listTags(): Promise<FavoriteTag[]> {
  const res = await apiFetchJson<Schemas["FavoriteTagListResponse"]>(apiUrl("v1/me/tags"));
  return res.tags;
}

export async function createTag(
  input: Schemas["TagCreate"],
): Promise<Schemas["FavoriteTagCreateResponse"]> {
  return apiFetchJson<Schemas["FavoriteTagCreateResponse"]>(
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
  input: Schemas["TagUpdate"],
): Promise<FavoriteTag> {
  const res = await apiFetchJson<Schemas["FavoriteTagResponse"]>(
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
  await apiFetchJson<Schemas["FavoriteDeletedResponse"]>(apiUrl(`v1/me/tags/${id}`), {
    method: "DELETE",
  });
}

export async function setFavoriteTags(
  favoriteId: string,
  tagIds: string[],
): Promise<FavoriteTag[]> {
  const res = await apiFetchJson<Schemas["FavoriteTagsResponse"]>(
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
  const res = await apiFetchJson<Schemas["FavoriteTagsResponse"]>(
    apiUrl(`v1/me/favorites/${favoriteId}/tags`),
  );
  return res.tags;
}
