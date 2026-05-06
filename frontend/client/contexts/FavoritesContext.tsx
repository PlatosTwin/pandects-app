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
import { useAuth } from "@/hooks/use-auth";
import {
  createTag as apiCreateTag,
  deleteFavorite as apiDeleteFavorite,
  deleteTag as apiDeleteTag,
  favoritesExists,
  listTags as apiListTags,
  patchFavorite as apiPatchFavorite,
  patchTag as apiPatchTag,
  setFavoriteTags as apiSetFavoriteTags,
  upsertFavorite as apiUpsertFavorite,
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
  ) => Promise<{ id: string; created: boolean }>;
  deleteFavorite: (
    itemType: FavoriteItemType,
    itemUuid: string,
    favoriteId: string,
  ) => Promise<void>;
  patchFavorite: (
    favoriteId: string,
    patch: { note?: string | null; project_id?: string | null },
  ) => Promise<void>;
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
  setFavoriteTagsCache: (favoriteId: string, tags: FavoriteTag[]) => void;
  setFavoriteTags: (
    favoriteId: string,
    tagIds: string[],
  ) => Promise<FavoriteTag[]>;
}

const FavoritesContext = createContext<FavoritesContextValue | null>(null);

export function FavoritesProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const userId = user?.id ?? null;

  const [byType, setByType] = useState<Record<FavoriteItemType, StarMap>>({
    section: {},
    agreement: {},
    tax_clause: {},
  });

  const [tags, setTags] = useState<FavoriteTag[]>([]);
  const [favoriteTagsMap, setFavoriteTagsMap] = useState<
    Record<string, FavoriteTag[]>
  >({});
  const tagsLoadedRef = useRef<boolean>(false);

  // Reset when user changes (sign-in/sign-out).
  const lastUserRef = useRef<string | null>(null);
  useEffect(() => {
    if (lastUserRef.current !== userId) {
      lastUserRef.current = userId;
      setByType({ section: {}, agreement: {}, tax_clause: {} });
      requestedRef.current = {
        section: new Set(),
        agreement: new Set(),
        tax_clause: new Set(),
      };
      setTags([]);
      setFavoriteTagsMap({});
      tagsLoadedRef.current = false;
    }
  }, [userId]);

  const requestedRef = useRef<Record<FavoriteItemType, Set<string>>>({
    section: new Set(),
    agreement: new Set(),
    tax_clause: new Set(),
  });

  const ensureLoaded = useCallback(
    (itemType: FavoriteItemType, itemUuids: string[]) => {
      if (!userId) return;
      const requested = requestedRef.current[itemType];
      const missing = itemUuids.filter((u) => u && !requested.has(u));
      if (missing.length === 0) return;
      missing.forEach((u) => requested.add(u));
      // Bulk-load existence; non-existence is recorded by absence.
      void favoritesExists(itemType, missing)
        .then((map) => {
          if (Object.keys(map).length === 0) return;
          setByType((prev) => ({
            ...prev,
            [itemType]: { ...prev[itemType], ...map },
          }));
        })
        .catch(() => {
          // Soft-fail: forget so a later visit can retry.
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

  const upsertFavorite = useCallback(async (input: FavoriteCreateInput) => {
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
    return { id: favorite.id, created };
  }, []);

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

  const reloadTags = useCallback(async () => {
    if (!userId) return;
    try {
      const fresh = await apiListTags();
      setTags(fresh);
      tagsLoadedRef.current = true;
    } catch {
      // Soft-fail; consumers can retry by calling reloadTags.
    }
  }, [userId]);

  const ensureTagsLoaded = useCallback(() => {
    if (!userId || tagsLoadedRef.current) return;
    tagsLoadedRef.current = true;
    void reloadTags();
  }, [userId, reloadTags]);

  const createTag = useCallback(async (name: string, color: TagColor) => {
    const { tag } = await apiCreateTag({ name, color });
    setTags((prev) => {
      if (prev.some((t) => t.id === tag.id)) return prev;
      return [...prev, tag].sort((a, b) => a.name.localeCompare(b.name));
    });
    return tag;
  }, []);

  const updateTag = useCallback(
    async (id: string, patch: { name?: string; color?: TagColor }) => {
      const updated = await apiPatchTag(id, patch);
      setTags((prev) =>
        prev
          .map((t) => (t.id === updated.id ? updated : t))
          .sort((a, b) => a.name.localeCompare(b.name)),
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
    [],
  );

  const removeTag = useCallback(async (id: string) => {
    await apiDeleteTag(id);
    setTags((prev) => prev.filter((t) => t.id !== id));
    setFavoriteTagsMap((prev) => {
      const next: Record<string, FavoriteTag[]> = {};
      for (const [favId, ts] of Object.entries(prev)) {
        next[favId] = ts.filter((t) => t.id !== id);
      }
      return next;
    });
  }, []);

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
      tags,
      ensureTagsLoaded,
      reloadTags,
      createTag,
      updateTag,
      removeTag,
      tagsForFavorite,
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
      tags,
      ensureTagsLoaded,
      reloadTags,
      createTag,
      updateTag,
      removeTag,
      tagsForFavorite,
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
