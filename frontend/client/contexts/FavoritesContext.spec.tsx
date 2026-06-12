// @vitest-environment jsdom
/**
 * Behavioral tests for FavoritesContext mutations: optimistic star/unstar
 * with rollback, existence-map dedupe, catalog mutations, and the
 * sign-out reset.
 */
import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Favorite, FavoriteTag } from "@/lib/favorites-api";

vi.mock("@/lib/favorites-api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/favorites-api")>();
  return {
    ...actual,
    favoritesExists: vi.fn(),
    upsertFavorite: vi.fn(),
    deleteFavorite: vi.fn(),
    patchFavorite: vi.fn(),
    listFavoriteProjects: vi.fn(),
    createProject: vi.fn(),
    patchProject: vi.fn(),
    deleteProject: vi.fn(),
    listTags: vi.fn(),
    createTag: vi.fn(),
    patchTag: vi.fn(),
    deleteTag: vi.fn(),
    getFavoriteTags: vi.fn(),
    setFavoriteTags: vi.fn(),
  };
});

const mockUseAuth = vi.fn();
vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => mockUseAuth(),
}));

import * as favoritesApi from "@/lib/favorites-api";
import {
  FavoritesProvider,
  useFavorites,
} from "@/contexts/FavoritesContext";

const api = vi.mocked(favoritesApi);

function makeFavorite(overrides: Partial<Favorite> = {}): Favorite {
  return {
    id: "fav-1",
    project_id: "proj-1",
    project_ids: ["proj-1"],
    item_type: "section",
    item_uuid: "uuid-1",
    agreement_uuid: null,
    note: null,
    context: null,
    tags: [],
    created_at: null,
    updated_at: null,
    ...overrides,
  };
}

function makeTag(overrides: Partial<FavoriteTag> = {}): FavoriteTag {
  return { id: "tag-1", name: "alpha", color: "slate", ...overrides };
}

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { refetchOnWindowFocus: false, retry: false, staleTime: 0 },
    },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(
      QueryClientProvider,
      { client: queryClient },
      createElement(FavoritesProvider, null, children),
    );
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockUseAuth.mockReturnValue({ user: { id: "user-1" } });
});

describe("FavoritesContext star flow", () => {
  it("upsertFavorite marks the item starred and caches its tags", async () => {
    api.upsertFavorite.mockResolvedValue({
      favorite: makeFavorite({ tags: [makeTag()] }),
      created: true,
    });
    const { result } = renderHook(() => useFavorites(), {
      wrapper: createWrapper(),
    });

    expect(result.current.isStarred("section", "uuid-1")).toBe(false);

    await act(async () => {
      await result.current.upsertFavorite({
        item_type: "section",
        item_uuid: "uuid-1",
      });
    });

    expect(result.current.isStarred("section", "uuid-1")).toBe(true);
    expect(result.current.favoriteIdFor("section", "uuid-1")).toBe("fav-1");
    expect(result.current.tagsForFavorite("fav-1")).toEqual([makeTag()]);
  });

  it("deleteFavorite unstars optimistically and rolls back on API failure", async () => {
    api.upsertFavorite.mockResolvedValue({
      favorite: makeFavorite({ tags: [makeTag()] }),
      created: true,
    });
    const { result } = renderHook(() => useFavorites(), {
      wrapper: createWrapper(),
    });
    await act(async () => {
      await result.current.upsertFavorite({
        item_type: "section",
        item_uuid: "uuid-1",
      });
    });

    // Success path: optimistic removal sticks.
    api.deleteFavorite.mockResolvedValueOnce(undefined);
    await act(async () => {
      await result.current.deleteFavorite("section", "uuid-1", "fav-1");
    });
    expect(result.current.isStarred("section", "uuid-1")).toBe(false);

    // Failure path: re-star, then fail the delete; star and tags come back.
    await act(async () => {
      await result.current.upsertFavorite({
        item_type: "section",
        item_uuid: "uuid-1",
      });
    });
    api.deleteFavorite.mockRejectedValueOnce(new Error("boom"));
    await expect(
      act(async () => {
        await result.current.deleteFavorite("section", "uuid-1", "fav-1");
      }),
    ).rejects.toThrow("boom");
    expect(result.current.isStarred("section", "uuid-1")).toBe(true);
    expect(result.current.favoriteIdFor("section", "uuid-1")).toBe("fav-1");
    expect(result.current.tagsForFavorite("fav-1")).toEqual([makeTag()]);
  });

  it("ensureLoaded requests each uuid once and re-requests after a failure", async () => {
    api.favoritesExists.mockResolvedValue({ "uuid-1": "fav-1" });
    const { result } = renderHook(() => useFavorites(), {
      wrapper: createWrapper(),
    });

    act(() => {
      result.current.ensureLoaded("section", ["uuid-1", "uuid-2"]);
    });
    await waitFor(() =>
      expect(result.current.isStarred("section", "uuid-1")).toBe(true),
    );
    expect(api.favoritesExists).toHaveBeenCalledTimes(1);
    expect(api.favoritesExists).toHaveBeenCalledWith("section", [
      "uuid-1",
      "uuid-2",
    ]);

    // Same uuids again: deduped, no extra request.
    act(() => {
      result.current.ensureLoaded("section", ["uuid-1", "uuid-2"]);
    });
    expect(api.favoritesExists).toHaveBeenCalledTimes(1);

    // A failing batch frees its uuids for a later retry.
    api.favoritesExists.mockRejectedValueOnce(new Error("offline"));
    act(() => {
      result.current.ensureLoaded("section", ["uuid-3"]);
    });
    await waitFor(() => expect(api.favoritesExists).toHaveBeenCalledTimes(2));
    api.favoritesExists.mockResolvedValueOnce({ "uuid-3": "fav-3" });
    act(() => {
      result.current.ensureLoaded("section", ["uuid-3"]);
    });
    await waitFor(() =>
      expect(result.current.isStarred("section", "uuid-3")).toBe(true),
    );
    expect(api.favoritesExists).toHaveBeenCalledTimes(3);
  });

  it("resets star state when the user signs out", async () => {
    api.upsertFavorite.mockResolvedValue({
      favorite: makeFavorite(),
      created: true,
    });
    const { result, rerender } = renderHook(() => useFavorites(), {
      wrapper: createWrapper(),
    });
    await act(async () => {
      await result.current.upsertFavorite({
        item_type: "section",
        item_uuid: "uuid-1",
      });
    });
    expect(result.current.isStarred("section", "uuid-1")).toBe(true);

    mockUseAuth.mockReturnValue({ user: null });
    rerender();
    await waitFor(() =>
      expect(result.current.isStarred("section", "uuid-1")).toBe(false),
    );
  });
});

describe("FavoritesContext catalogs", () => {
  it("createTag inserts into the sorted tag catalog; removeTag also strips per-favorite caches", async () => {
    api.listTags.mockResolvedValue([makeTag({ id: "tag-b", name: "beta" })]);
    api.upsertFavorite.mockResolvedValue({
      favorite: makeFavorite({
        tags: [makeTag({ id: "tag-b", name: "beta" })],
      }),
      created: true,
    });
    const { result } = renderHook(() => useFavorites(), {
      wrapper: createWrapper(),
    });

    act(() => {
      result.current.ensureTagsLoaded();
    });
    await waitFor(() => expect(result.current.tags).toHaveLength(1));

    api.createTag.mockResolvedValue({
      tag: makeTag({ id: "tag-a", name: "alpha" }),
      created: true,
    });
    await act(async () => {
      await result.current.createTag("alpha", "slate");
    });
    // Catalog updates land via the React Query cache, whose observer
    // notifications are batched asynchronously — hence waitFor.
    await waitFor(() =>
      expect(result.current.tags.map((t) => t.name)).toEqual(["alpha", "beta"]),
    );

    await act(async () => {
      await result.current.upsertFavorite({
        item_type: "section",
        item_uuid: "uuid-1",
      });
    });
    expect(result.current.tagsForFavorite("fav-1")).toHaveLength(1);

    api.deleteTag.mockResolvedValue(undefined);
    await act(async () => {
      await result.current.removeTag("tag-b");
    });
    await waitFor(() =>
      expect(result.current.tags.map((t) => t.name)).toEqual(["alpha"]),
    );
    expect(result.current.tagsForFavorite("fav-1")).toEqual([]);
  });

  it("project create/update/remove keep the catalog sorted and consistent", async () => {
    api.listFavoriteProjects.mockResolvedValue([
      {
        id: "proj-1",
        name: "Default",
        color: "slate",
        sort_order: 0,
        is_default: true,
      } as favoritesApi.FavoriteProject,
    ]);
    const { result } = renderHook(() => useFavorites(), {
      wrapper: createWrapper(),
    });
    act(() => {
      result.current.ensureProjectsLoaded();
    });
    await waitFor(() => expect(result.current.projects).toHaveLength(1));

    api.createProject.mockResolvedValue({
      id: "proj-2",
      name: "Deals",
      color: "slate",
      sort_order: 1,
      is_default: false,
    } as favoritesApi.FavoriteProject);
    await act(async () => {
      await result.current.createProject("Deals", "slate");
    });
    await waitFor(() =>
      expect(result.current.projects.map((p) => p.id)).toEqual([
        "proj-1",
        "proj-2",
      ]),
    );

    api.patchProject.mockResolvedValue({
      id: "proj-2",
      name: "Renamed",
      color: "slate",
      sort_order: 1,
      is_default: false,
    } as favoritesApi.FavoriteProject);
    await act(async () => {
      await result.current.updateProject("proj-2", { name: "Renamed" });
    });
    await waitFor(() =>
      expect(result.current.projects[1]?.name).toBe("Renamed"),
    );

    api.deleteProject.mockResolvedValue({
      reassigned_to_project_id: "proj-1",
      moved: 2,
    } as Awaited<ReturnType<typeof favoritesApi.deleteProject>>);
    await act(async () => {
      await result.current.removeProject("proj-2", "proj-1");
    });
    await waitFor(() =>
      expect(result.current.projects.map((p) => p.id)).toEqual(["proj-1"]),
    );
  });
});
