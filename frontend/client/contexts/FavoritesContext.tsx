import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/hooks/use-auth";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { keys } from "@/lib/query-keys";
import {
  createProject as apiCreateProject,
  createTag as apiCreateTag,
  deleteFavorite as apiDeleteFavorite,
  deleteProject as apiDeleteProject,
  deleteTag as apiDeleteTag,
  favoritesExists,
  getFavoriteTags as apiGetFavoriteTags,
  listFavoriteProjects as apiListFavoriteProjects,
  listTags as apiListTags,
  patchFavorite as apiPatchFavorite,
  patchProject as apiPatchProject,
  patchTag as apiPatchTag,
  setFavoriteTags as apiSetFavoriteTags,
  upsertFavorite as apiUpsertFavorite,
  type FavoriteProject,
  type FavoriteCreateInput,
  type FavoriteItemType,
  type FavoriteTag,
  type TagColor,
} from "@/lib/favorites-api";

type StarMap = Record<string, string>;

interface FavoritesContextValue {
  isStarred: (itemType: FavoriteItemType, itemUuid: string) => boolean;
  favoriteIdFor: (
    itemType: FavoriteItemType,
    itemUuid: string,
  ) => string | null;
  ensureLoaded: (itemType: FavoriteItemType, itemUuids: string[]) => void;
  upsertFavorite: (
    input: FavoriteCreateInput,
  ) => Promise<{ id: string; project_id: string; created: boolean }>;
  deleteFavorite: (
    itemType: FavoriteItemType,
    itemUuid: string,
    favoriteId: string,
  ) => Promise<void>;
  patchFavorite: (
    favoriteId: string,
    patch: { note?: string | null; project_id?: string | null },
  ) => Promise<void>;
  // Project catalog
  projects: FavoriteProject[];
  ensureProjectsLoaded: () => void;
  reloadProjects: () => Promise<FavoriteProject[]>;
  createProject: (name: string, color: TagColor) => Promise<FavoriteProject>;
  updateProject: (
    id: string,
    patch: { name?: string; color?: TagColor; sort_order?: number },
  ) => Promise<FavoriteProject>;
  removeProject: (
    id: string,
    reassignProjectId?: string,
  ) => Promise<{ reassigned_to_project_id: string; moved: number }>;
  // Tag catalog
  tags: FavoriteTag[];
  ensureTagsLoaded: () => void;
  reloadTags: () => Promise<void>;
  createTag: (name: string, color: TagColor) => Promise<FavoriteTag>;
  updateTag: (
    id: string,
    patch: { name?: string; color?: TagColor },
  ) => Promise<FavoriteTag>;
  removeTag: (id: string) => Promise<void>;
  // Per-favorite tag cache
  tagsForFavorite: (favoriteId: string) => FavoriteTag[] | undefined;
  loadTagsForFavorite: (favoriteId: string) => Promise<FavoriteTag[]>;
  setFavoriteTagsCache: (favoriteId: string, tags: FavoriteTag[]) => void;
  setFavoriteTags: (
    favoriteId: string,
    tagIds: string[],
  ) => Promise<FavoriteTag[]>;
}

const FavoritesContext = createContext<FavoritesContextValue | null>(null);

const sortProjects = (list: FavoriteProject[]) =>
  [...list].sort((a, b) => a.sort_order - b.sort_order);
const sortTags = (list: FavoriteTag[]) =>
  [...list].sort((a, b) => a.name.localeCompare(b.name));

export function FavoritesProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const userId = user?.id ?? null;
  const queryClient = useQueryClient();

  // Lazy existence map: keyed by (itemType, itemUuid) → favoriteId. The model
  // is "incremental, request-on-demand for a specific uuid set" — not a great
  // fit for one RQ key per uuid, so we keep it as local state and dedupe
  // requested uuids via a ref.
  const [byType, setByType] = useState<Record<FavoriteItemType, StarMap>>({
    section: {},
    agreement: {},
    tax_clause: {},
  });
  const requestedRef = useRef<Record<FavoriteItemType, Set<string>>>({
    section: new Set(),
    agreement: new Set(),
    tax_clause: new Set(),
  });

  // Per-favorite tag cache. Same shape rationale as byType.
  const [favoriteTagsMap, setFavoriteTagsMap] = useState<
    Record<string, FavoriteTag[]>
  >({});

  // Lazy-enable gates for catalogs: caller signals interest via
  // ensureTagsLoaded/ensureProjectsLoaded, and useQuery picks up the work.
  const [tagsEnabled, setTagsEnabled] = useState(false);
  const [projectsEnabled, setProjectsEnabled] = useState(false);

  // Reset all state on sign-in/sign-out.
  const lastUserRef = useRef<string | null>(null);
  useEffect(() => {
    if (lastUserRef.current === userId) return;
    lastUserRef.current = userId;
    setByType({ section: {}, agreement: {}, tax_clause: {} });
    requestedRef.current = {
      section: new Set(),
      agreement: new Set(),
      tax_clause: new Set(),
    };
    setFavoriteTagsMap({});
    setTagsEnabled(false);
    setProjectsEnabled(false);
    queryClient.removeQueries({ queryKey: keys.favorites.all });
  }, [userId, queryClient]);

  // --- Catalogs via React Query ---
  const tagsQuery = useQuery({
    queryKey: keys.favorites.tags,
    queryFn: async () => sortTags(await apiListTags()),
    enabled: !!userId && tagsEnabled && !IS_SERVER_RENDER,
    staleTime: 5 * 60 * 1000,
  });
  const tags = tagsQuery.data ?? [];

  const projectsQuery = useQuery({
    queryKey: keys.favorites.projects,
    queryFn: async () => sortProjects(await apiListFavoriteProjects()),
    enabled: !!userId && projectsEnabled && !IS_SERVER_RENDER,
    staleTime: 5 * 60 * 1000,
  });
  const projects = projectsQuery.data ?? [];

  // --- Existence map ---
  const ensureLoaded = useCallback(
    (itemType: FavoriteItemType, itemUuids: string[]) => {
      if (!userId) return;
      const requested = requestedRef.current[itemType];
      const missing = itemUuids.filter((u) => u && !requested.has(u));
      if (missing.length === 0) return;
      missing.forEach((u) => requested.add(u));
      void favoritesExists(itemType, missing)
        .then((map) => {
          if (Object.keys(map).length === 0) return;
          setByType((prev) => ({
            ...prev,
            [itemType]: { ...prev[itemType], ...map },
          }));
        })
        .catch(() => {
          missing.forEach((u) => requested.delete(u));
        });
    },
    [userId],
  );

  const isStarred = useCallback(
    (itemType: FavoriteItemType, itemUuid: string) =>
      Boolean(byType[itemType][itemUuid]),
    [byType],
  );

  const favoriteIdFor = useCallback(
    (itemType: FavoriteItemType, itemUuid: string) =>
      byType[itemType][itemUuid] ?? null,
    [byType],
  );

  // --- Favorite mutations ---
  const upsertFavorite = useCallback(
    async (input: FavoriteCreateInput) => {
      const { favorite, created } = await apiUpsertFavorite(input);
      setByType((prev) => ({
        ...prev,
        [favorite.item_type]: {
          ...prev[favorite.item_type],
          [favorite.item_uuid]: favorite.id,
        },
      }));
      setFavoriteTagsMap((prev) => ({
        ...prev,
        [favorite.id]: favorite.tags ?? [],
      }));
      return { id: favorite.id, project_id: favorite.project_id, created };
    },
    [],
  );

  const deleteFavorite = useCallback(
    async (
      itemType: FavoriteItemType,
      itemUuid: string,
      favoriteId: string,
    ) => {
      const previousTags = favoriteTagsMap[favoriteId];
      setByType((prev) => {
        const next = { ...prev[itemType] };
        delete next[itemUuid];
        return { ...prev, [itemType]: next };
      });
      setFavoriteTagsMap((prev) => {
        const next = { ...prev };
        delete next[favoriteId];
        return next;
      });
      try {
        await apiDeleteFavorite(favoriteId);
      } catch (error) {
        setByType((prev) => ({
          ...prev,
          [itemType]: {
            ...prev[itemType],
            [itemUuid]: favoriteId,
          },
        }));
        if (previousTags) {
          setFavoriteTagsMap((prev) => ({
            ...prev,
            [favoriteId]: previousTags,
          }));
        }
        throw error;
      }
    },
    [favoriteTagsMap],
  );

  const patchFavorite = useCallback(
    async (
      favoriteId: string,
      patch: { note?: string | null; project_id?: string | null },
    ) => {
      await apiPatchFavorite(favoriteId, patch);
    },
    [],
  );

  // --- Projects ---
  const ensureProjectsLoaded = useCallback(() => {
    if (!userId) return;
    setProjectsEnabled(true);
  }, [userId]);

  const reloadProjects = useCallback(async () => {
    if (!userId) return [];
    setProjectsEnabled(true);
    const fresh = sortProjects(await apiListFavoriteProjects());
    queryClient.setQueryData<FavoriteProject[]>(keys.favorites.projects, fresh);
    return fresh;
  }, [userId, queryClient]);

  const createProject = useCallback(
    async (name: string, color: TagColor) => {
      const project = await apiCreateProject({ name, color });
      queryClient.setQueryData<FavoriteProject[]>(
        keys.favorites.projects,
        (prev) => sortProjects([...(prev ?? []), project]),
      );
      return project;
    },
    [queryClient],
  );

  const updateProject = useCallback(
    async (
      id: string,
      patch: { name?: string; color?: TagColor; sort_order?: number },
    ) => {
      const updated = await apiPatchProject(id, patch);
      queryClient.setQueryData<FavoriteProject[]>(
        keys.favorites.projects,
        (prev) =>
          sortProjects(
            (prev ?? []).map((p) => (p.id === updated.id ? updated : p)),
          ),
      );
      return updated;
    },
    [queryClient],
  );

  const removeProject = useCallback(
    async (id: string, reassignProjectId?: string) => {
      const result = await apiDeleteProject(id, reassignProjectId);
      queryClient.setQueryData<FavoriteProject[]>(
        keys.favorites.projects,
        (prev) => (prev ?? []).filter((p) => p.id !== id),
      );
      return {
        reassigned_to_project_id: result.reassigned_to_project_id,
        moved: result.moved,
      };
    },
    [queryClient],
  );

  // --- Tags ---
  const ensureTagsLoaded = useCallback(() => {
    if (!userId) return;
    setTagsEnabled(true);
  }, [userId]);

  const reloadTags = useCallback(async () => {
    if (!userId) return;
    setTagsEnabled(true);
    try {
      const fresh = sortTags(await apiListTags());
      queryClient.setQueryData<FavoriteTag[]>(keys.favorites.tags, fresh);
    } catch {
      // Soft-fail; consumers can retry by calling reloadTags.
    }
  }, [userId, queryClient]);

  const createTag = useCallback(
    async (name: string, color: TagColor) => {
      const { tag } = await apiCreateTag({ name, color });
      queryClient.setQueryData<FavoriteTag[]>(keys.favorites.tags, (prev) => {
        const list = prev ?? [];
        if (list.some((t) => t.id === tag.id)) return list;
        return sortTags([...list, tag]);
      });
      return tag;
    },
    [queryClient],
  );

  const updateTag = useCallback(
    async (id: string, patch: { name?: string; color?: TagColor }) => {
      const updated = await apiPatchTag(id, patch);
      queryClient.setQueryData<FavoriteTag[]>(keys.favorites.tags, (prev) =>
        sortTags(
          (prev ?? []).map((t) => (t.id === updated.id ? updated : t)),
        ),
      );
      setFavoriteTagsMap((prev) => {
        const next: Record<string, FavoriteTag[]> = {};
        for (const [favoriteId, favoriteTags] of Object.entries(prev)) {
          next[favoriteId] = favoriteTags.map((tag) =>
            tag.id === updated.id ? updated : tag,
          );
        }
        return next;
      });
      return updated;
    },
    [queryClient],
  );

  const removeTag = useCallback(
    async (id: string) => {
      await apiDeleteTag(id);
      queryClient.setQueryData<FavoriteTag[]>(keys.favorites.tags, (prev) =>
        (prev ?? []).filter((t) => t.id !== id),
      );
      setFavoriteTagsMap((prev) => {
        const next: Record<string, FavoriteTag[]> = {};
        for (const [favId, ts] of Object.entries(prev)) {
          next[favId] = ts.filter((t) => t.id !== id);
        }
        return next;
      });
    },
    [queryClient],
  );

  // --- Per-favorite tags ---
  const tagsForFavorite = useCallback(
    (favoriteId: string) => favoriteTagsMap[favoriteId],
    [favoriteTagsMap],
  );

  const setFavoriteTagsCache = useCallback(
    (favoriteId: string, list: FavoriteTag[]) => {
      setFavoriteTagsMap((prev) => ({ ...prev, [favoriteId]: list }));
    },
    [],
  );

  const loadTagsForFavorite = useCallback(async (favoriteId: string) => {
    const fresh = await apiGetFavoriteTags(favoriteId);
    setFavoriteTagsMap((prev) => ({ ...prev, [favoriteId]: fresh }));
    return fresh;
  }, []);

  const setFavoriteTags = useCallback(
    async (favoriteId: string, tagIds: string[]) => {
      const fresh = await apiSetFavoriteTags(favoriteId, tagIds);
      setFavoriteTagsMap((prev) => ({ ...prev, [favoriteId]: fresh }));
      return fresh;
    },
    [],
  );

  const value = useMemo<FavoritesContextValue>(
    () => ({
      isStarred,
      favoriteIdFor,
      ensureLoaded,
      upsertFavorite,
      deleteFavorite,
      patchFavorite,
      projects,
      ensureProjectsLoaded,
      reloadProjects,
      createProject,
      updateProject,
      removeProject,
      tags,
      ensureTagsLoaded,
      reloadTags,
      createTag,
      updateTag,
      removeTag,
      tagsForFavorite,
      loadTagsForFavorite,
      setFavoriteTagsCache,
      setFavoriteTags,
    }),
    [
      isStarred,
      favoriteIdFor,
      ensureLoaded,
      upsertFavorite,
      deleteFavorite,
      patchFavorite,
      projects,
      ensureProjectsLoaded,
      reloadProjects,
      createProject,
      updateProject,
      removeProject,
      tags,
      ensureTagsLoaded,
      reloadTags,
      createTag,
      updateTag,
      removeTag,
      tagsForFavorite,
      loadTagsForFavorite,
      setFavoriteTagsCache,
      setFavoriteTags,
    ],
  );

  return (
    <FavoritesContext.Provider value={value}>
      {children}
    </FavoritesContext.Provider>
  );
}

export function useFavorites(): FavoritesContextValue {
  const ctx = useContext(FavoritesContext);
  if (!ctx) {
    throw new Error("useFavorites must be used within <FavoritesProvider />");
  }
  return ctx;
}
