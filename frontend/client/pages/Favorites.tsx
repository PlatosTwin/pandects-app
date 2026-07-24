import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import { sortableKeyboardCoordinates } from "@dnd-kit/sortable";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Star } from "lucide-react";

import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { BulkTagActions } from "@/components/favorites/BulkTagActions";
import { FavoriteRow } from "@/components/favorites/FavoriteRow";
import { FilterBar } from "@/components/favorites/FilterBar";
import { ProjectSidebar } from "@/components/favorites/ProjectSidebar";
import { TagsManager } from "@/components/favorites/TagsManager";
import {
  fetchAgreementMetadataBatch,
  fetchSectionDetailsBatch,
} from "@/components/favorites/api";
import {
  EMPTY_FILTERS,
  type AgreementMetadata,
  type FavoriteFilters,
  type Filter,
  type SectionDetails,
} from "@/components/favorites/types";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/use-auth";
import { useFavorites } from "@/contexts/FavoritesContext";
import { indexClauseTypeLabels } from "@/lib/clause-type-index";
import { useFilterOptions } from "@/hooks/use-filter-options";
import {
  bulkCopyFavorites as apiBulkCopyFavorites,
  bulkMoveFavorites as apiBulkMoveFavorites,
  bulkUpdateFavoriteTags as apiBulkUpdateFavoriteTags,
  listFavorites,
  type Favorite,
  type FavoriteProject,
  type FavoriteTag,
} from "@/lib/favorites-api";

const EMPTY_AGREEMENTS: Record<string, AgreementMetadata | null> = {};
const EMPTY_SECTIONS: Record<string, SectionDetails | null> = {};

export default function FavoritesPage() {
  const { status, user } = useAuth();
  const { toast } = useToast();
  const { clause_types } = useFilterOptions({ fields: ["clause_types"] });
  const clauseTypeLabelById = useMemo(
    () => indexClauseTypeLabels(clause_types),
    [clause_types],
  );
  const {
    projects,
    ensureProjectsLoaded,
    reloadProjects,
    setFavoriteTagsCache,
  } = useFavorites();
  const [filter, setFilter] = useState<Filter>("all");
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [favorites, setFavorites] = useState<Favorite[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState<FavoriteFilters>(EMPTY_FILTERS);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkProjectId, setBulkProjectId] = useState("");
  const [bulkCopyProjectId, setBulkCopyProjectId] = useState("");
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );

  useEffect(() => {
    if (status !== "authenticated" || !user) return;
    ensureProjectsLoaded();
    setLoading(true);
    listFavorites()
      .then((rows) => {
        const normalized = rows.map((row) => ({
          ...row,
          project_ids:
            row.project_ids.length > 0 ? row.project_ids : [row.project_id],
        }));
        setFavorites(normalized);
        for (const fav of normalized) {
          setFavoriteTagsCache(fav.id, fav.tags);
        }
      })
      .catch(() => {
        toast({
          title: "Couldn't load favorites",
          variant: "destructive",
        });
      })
      .finally(() => setLoading(false));
  }, [status, user, toast, ensureProjectsLoaded, setFavoriteTagsCache]);

  // One batched request for agreement metadata (filtering + richer display)
  // instead of one request per favorite.
  const agreementUuids = useMemo(() => {
    const unique = new Set<string>();
    for (const fav of favorites) {
      if (fav.agreement_uuid) unique.add(fav.agreement_uuid);
    }
    return Array.from(unique).sort();
  }, [favorites]);
  const { data: agreementByUuid = EMPTY_AGREEMENTS } = useQuery({
    queryKey: ["favorites-agreement-metadata", agreementUuids],
    queryFn: () => fetchAgreementMetadataBatch(agreementUuids),
    enabled:
      status === "authenticated" && !!user && agreementUuids.length > 0,
    staleTime: 5 * 60 * 1000,
  });

  // Same batching for section details: one request per 500 section favorites.
  // While the query is in flight, lookups are undefined (= loading); after it
  // settles, misses are null (= "Details unavailable").
  const sectionUuids = useMemo(() => {
    const unique = new Set<string>();
    for (const fav of favorites) {
      if (fav.item_type === "section") unique.add(fav.item_uuid);
    }
    return Array.from(unique).sort();
  }, [favorites]);
  const { data: sectionByUuid = EMPTY_SECTIONS } = useQuery({
    queryKey: ["favorites-section-details", sectionUuids],
    queryFn: () => fetchSectionDetailsBatch(sectionUuids),
    enabled: status === "authenticated" && !!user && sectionUuids.length > 0,
    staleTime: 5 * 60 * 1000,
  });

  const counts = useMemo(() => {
    const c: Record<Filter, number> = {
      all: favorites.length,
      section: 0,
      agreement: 0,
      tax_clause: 0,
    };
    for (const fav of favorites) c[fav.item_type] += 1;
    return c;
  }, [favorites]);

  const projectCounts = useMemo(() => {
    const next: Record<string, number> = {};
    for (const fav of favorites) {
      const projectIds =
        fav.project_ids.length > 0 ? fav.project_ids : [fav.project_id];
      for (const projectId of projectIds) {
        next[projectId] = (next[projectId] ?? 0) + 1;
      }
    }
    return next;
  }, [favorites]);

  const visible = useMemo(() => {
    const yearMin = filters.yearMin.trim() ? Number(filters.yearMin) : null;
    const yearMax = filters.yearMax.trim() ? Number(filters.yearMax) : null;
    const sizeMin = filters.sizeMinUsd.trim()
      ? Number(filters.sizeMinUsd)
      : null;
    const sizeMax = filters.sizeMaxUsd.trim()
      ? Number(filters.sizeMaxUsd)
      : null;
    const targetQ = filters.target.trim().toLowerCase();
    const acquirerQ = filters.acquirer.trim().toLowerCase();

    return favorites.filter((fav) => {
      if (
        activeProjectId !== null &&
        !fav.project_ids.includes(activeProjectId) &&
        fav.project_id !== activeProjectId
      ) {
        return false;
      }
      if (filter !== "all" && fav.item_type !== filter) return false;
      if (filters.tagIds.length > 0) {
        const ids = new Set(fav.tags.map((t) => t.id));
        if (!filters.tagIds.every((id) => ids.has(id))) return false;
      }

      const agreement = fav.agreement_uuid
        ? (agreementByUuid[fav.agreement_uuid] ?? null)
        : null;
      const ctxTarget = (
        agreement?.target ??
        (typeof fav.context?.target === "string"
          ? (fav.context.target as string)
          : "")
      )
        .toString()
        .toLowerCase();
      const ctxAcquirer = (
        agreement?.acquirer ??
        (typeof fav.context?.acquirer === "string"
          ? (fav.context.acquirer as string)
          : "")
      )
        .toString()
        .toLowerCase();

      if (targetQ && !ctxTarget.includes(targetQ)) return false;
      if (acquirerQ && !ctxAcquirer.includes(acquirerQ)) return false;

      // Year + size filters require agreement metadata to be loaded; if it
      // hasn't loaded yet, exclude when the user has set those filters.
      if (yearMin !== null || yearMax !== null) {
        const year = agreement?.year ?? null;
        if (year === null) return false;
        if (yearMin !== null && year < yearMin) return false;
        if (yearMax !== null && year > yearMax) return false;
      }
      if (sizeMin !== null || sizeMax !== null) {
        const size = agreement?.transaction_price_total ?? null;
        if (size === null) return false;
        if (sizeMin !== null && size < sizeMin) return false;
        if (sizeMax !== null && size > sizeMax) return false;
      }
      return true;
    });
  }, [activeProjectId, filter, filters, favorites, agreementByUuid]);

  useEffect(() => {
    setSelectedIds((prev) => {
      if (prev.size === 0) return prev;
      const visibleIds = new Set(visible.map((fav) => fav.id));
      const next = new Set(Array.from(prev).filter((id) => visibleIds.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [visible]);

  if (status === "loading" || !user) {
    return (
      <PageShell title="Favorites">
        <div className="text-sm text-muted-foreground">Loading…</div>
      </PageShell>
    );
  }

  const handleUpdated = (next: Favorite) =>
    setFavorites((prev) => prev.map((f) => (f.id === next.id ? next : f)));
  const handleRemoved = (id: string) =>
    setFavorites((prev) => prev.filter((f) => f.id !== id));
  const handleTagsChanged = (id: string, tags: FavoriteTag[]) => {
    setFavoriteTagsCache(id, tags);
    setFavorites((prev) => prev.map((f) => (f.id === id ? { ...f, tags } : f)));
  };
  const handleTagUpdated = (tag: FavoriteTag) => {
    setFavorites((prev) =>
      prev.map((f) => ({
        ...f,
        tags: f.tags.map((existing) =>
          existing.id === tag.id ? tag : existing,
        ),
      })),
    );
  };
  const handleTagDeleted = (tagId: string) => {
    setFilters((prev) => ({
      ...prev,
      tagIds: prev.tagIds.filter((id) => id !== tagId),
    }));
    setFavorites((prev) =>
      prev.map((f) => ({
        ...f,
        tags: f.tags.filter((tag) => tag.id !== tagId),
      })),
    );
  };
  const handleSelectFavorite = (id: string, selected: boolean) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (selected) {
        next.add(id);
      } else {
        next.delete(id);
      }
      return next;
    });
  };
  const visibleIds = visible.map((fav) => fav.id);
  const selectedVisibleCount = visibleIds.filter((id) =>
    selectedIds.has(id),
  ).length;
  const allVisibleSelected =
    visibleIds.length > 0 && selectedVisibleCount === visibleIds.length;
  const someVisibleSelected =
    selectedVisibleCount > 0 && selectedVisibleCount < visibleIds.length;
  const handleToggleVisibleSelection = (selected: boolean) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      for (const id of visibleIds) {
        if (selected) {
          next.add(id);
        } else {
          next.delete(id);
        }
      }
      return next;
    });
  };
  const handleMoveFavorites = async (
    favoriteIds: string[],
    projectId: string,
  ) => {
    if (favoriteIds.length === 0 || !projectId) return;
    try {
      const result = await apiBulkMoveFavorites(favoriteIds, projectId);
      const moved = new Set(result.favorite_ids);
      setFavorites((prev) =>
        prev.map((fav) =>
          moved.has(fav.id)
            ? {
                ...fav,
                project_id: result.project_id,
                project_ids: [result.project_id],
              }
            : fav,
        ),
      );
      setSelectedIds((prev) => {
        const next = new Set(prev);
        for (const id of moved) next.delete(id);
        return next;
      });
      setBulkProjectId("");
      const projectName =
        projects.find((project) => project.id === result.project_id)?.name ??
        "project";
      toast({ title: `Moved ${moved.size} to ${projectName}` });
    } catch {
      toast({ title: "Couldn't move favorites", variant: "destructive" });
    }
  };
  const handleCopyFavorites = async (
    favoriteIds: string[],
    projectId: string,
  ) => {
    if (favoriteIds.length === 0 || !projectId) return;
    try {
      const result = await apiBulkCopyFavorites(favoriteIds, [projectId]);
      const copied = new Set(result.favorite_ids);
      setFavorites((prev) =>
        prev.map((fav) =>
          copied.has(fav.id)
            ? {
                ...fav,
                project_ids: Array.from(
                  new Set([...fav.project_ids, ...result.project_ids]),
                ),
              }
            : fav,
        ),
      );
      setSelectedIds((prev) => {
        const next = new Set(prev);
        for (const id of copied) next.delete(id);
        return next;
      });
      setBulkCopyProjectId("");
      const projectName =
        projects.find((project) => project.id === projectId)?.name ?? "project";
      toast({ title: `Copied ${copied.size} to ${projectName}` });
    } catch {
      toast({ title: "Couldn't copy favorites", variant: "destructive" });
    }
  };
  const handleBulkUpdateTags = async (
    tagIds: string[],
    action: "add" | "remove",
  ) => {
    if (selectedIds.size === 0 || tagIds.length === 0) return;
    try {
      const result = await apiBulkUpdateFavoriteTags(
        Array.from(selectedIds),
        tagIds,
        action,
      );
      for (const [favoriteId, tags] of Object.entries(
        result.tags_by_favorite,
      )) {
        setFavoriteTagsCache(favoriteId, tags);
      }
      setFavorites((prev) =>
        prev.map((fav) => {
          const freshTags = result.tags_by_favorite[fav.id];
          if (!freshTags) return fav;
          return { ...fav, tags: freshTags };
        }),
      );
      toast({
        title:
          action === "add"
            ? "Tags applied to selected favorites"
            : "Tags removed from selected favorites",
      });
    } catch {
      toast({ title: "Couldn't update tags", variant: "destructive" });
      throw new Error("bulk tag update failed");
    }
  };
  const handleDragEnd = (event: DragEndEvent) => {
    const favoriteId = event.active.data.current?.favoriteId;
    const projectId = event.over?.data.current?.projectId;
    if (typeof favoriteId !== "string" || typeof projectId !== "string") return;
    const fav = favorites.find((row) => row.id === favoriteId);
    if (!fav || fav.project_id === projectId) return;
    void handleMoveFavorites([favoriteId], projectId);
  };
  const handleProjectCreated = (project: FavoriteProject) => {
    void reloadProjects();
    setActiveProjectId(project.id);
  };
  const handleProjectUpdated = (_project: FavoriteProject) => {
    void reloadProjects();
  };
  const handleProjectDeleted = (
    projectId: string,
    reassignProjectId: string,
  ) => {
    setFavorites((prev) =>
      prev.map((fav) =>
        fav.project_ids.includes(projectId) || fav.project_id === projectId
          ? {
              ...fav,
              project_id:
                fav.project_id === projectId
                  ? reassignProjectId
                  : fav.project_id,
              project_ids: Array.from(
                new Set(
                  fav.project_ids
                    .filter((id) => id !== projectId)
                    .concat(reassignProjectId),
                ),
              ),
            }
          : fav,
      ),
    );
    if (activeProjectId === projectId) {
      setActiveProjectId(reassignProjectId);
    }
    void reloadProjects();
  };

  return (
    <PageShell size="full" className="px-0 py-0 sm:px-0 lg:px-0">
      <DndContext sensors={sensors} onDragEnd={handleDragEnd}>
        <div className="w-full overflow-x-hidden">
          <div className="border-b border-border px-4 py-3 sm:px-8">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:gap-x-5">
                <h1 className="shrink-0 text-xl font-semibold tracking-tight text-foreground">
                  Favorites
                </h1>
              </div>
              <span className="text-sm text-muted-foreground">
                {visible.length} of {favorites.length} shown
              </span>
            </div>
          </div>

          <div className="border-b border-border bg-muted/20 px-4 py-2.5 backdrop-blur supports-[backdrop-filter]:bg-muted/20 sm:px-8">
            <div className="grid min-w-0 gap-2 xl:grid-cols-[18rem_minmax(0,1fr)]">
              <TagsManager
                onTagUpdated={handleTagUpdated}
                onTagDeleted={handleTagDeleted}
              />
              <FilterBar
                filters={filters}
                onChange={setFilters}
                onClear={() => setFilters(EMPTY_FILTERS)}
              />
            </div>
          </div>

          <div className="grid gap-4 px-4 py-4 sm:px-8 xl:grid-cols-[18rem_minmax(0,1fr)]">
            <ProjectSidebar
              projects={projects}
              activeProjectId={activeProjectId}
              projectCounts={projectCounts}
              onSelectProject={setActiveProjectId}
              onProjectCreated={handleProjectCreated}
              onProjectUpdated={handleProjectUpdated}
              onProjectDeleted={handleProjectDeleted}
            />

            <div className="min-w-0 space-y-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex min-w-0 flex-col gap-3 lg:flex-row lg:items-center">
                  <div className="shrink-0 text-sm font-semibold text-foreground">
                    Saved items
                  </div>
                  <Tabs
                    value={filter}
                    onValueChange={(value) => setFilter(value as Filter)}
                    className="min-w-0"
                  >
                    <div className="max-w-full overflow-x-auto">
                      {/* aria-controls overrides Radix's generated content ids:
                          all four tabs filter the one shared items panel. */}
                      <TabsList className="min-h-9 min-w-max border bg-background px-1 shadow-sm">
                        <TabsTrigger
                          className="px-3 text-sm"
                          value="all"
                          aria-controls="favorites-items-panel"
                        >
                          All ({counts.all})
                        </TabsTrigger>
                        <TabsTrigger
                          className="px-3 text-sm"
                          value="section"
                          aria-controls="favorites-items-panel"
                        >
                          Sections ({counts.section})
                        </TabsTrigger>
                        <TabsTrigger
                          className="px-3 text-sm"
                          value="agreement"
                          aria-controls="favorites-items-panel"
                        >
                          Deals ({counts.agreement})
                        </TabsTrigger>
                        <TabsTrigger
                          className="px-3 text-sm"
                          value="tax_clause"
                          aria-controls="favorites-items-panel"
                        >
                          Tax clauses ({counts.tax_clause})
                        </TabsTrigger>
                      </TabsList>
                    </div>
                  </Tabs>
                </div>
                <label className="flex min-h-9 items-center gap-2 rounded-md border bg-background px-3 text-sm font-medium text-foreground shadow-sm">
                  <Checkbox
                    checked={
                      allVisibleSelected
                        ? true
                        : someVisibleSelected
                          ? "indeterminate"
                          : false
                    }
                    onCheckedChange={(checked) =>
                      handleToggleVisibleSelection(checked === true)
                    }
                    disabled={visibleIds.length === 0}
                    aria-label={
                      allVisibleSelected
                        ? "Deselect all visible favorites"
                        : "Select all visible favorites"
                    }
                  />
                  <span>
                    {allVisibleSelected ? "Deselect all" : "Select all"}
                  </span>
                </label>
              </div>

              {selectedIds.size > 0 ? (
                <div className="sticky top-16 z-20 flex flex-col gap-2 rounded-lg border bg-background/95 px-3 py-2 shadow-sm backdrop-blur sm:flex-row sm:items-center">
                  <span className="text-sm font-medium text-foreground">
                    {selectedIds.size} selected
                  </span>
                  <div className="flex flex-wrap items-center gap-2">
                    <Select
                      value={bulkProjectId}
                      onValueChange={setBulkProjectId}
                    >
                      <SelectTrigger
                        className="h-11 w-full text-sm sm:h-8 sm:w-44"
                        aria-label="Move selected favorites to project"
                      >
                        <SelectValue placeholder="Move to project" />
                      </SelectTrigger>
                      <SelectContent>
                        {projects.map((project) => (
                          <SelectItem key={project.id} value={project.id}>
                            {project.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Button
                      type="button"
                      size="sm"
                      className="h-11 sm:h-8"
                      disabled={!bulkProjectId}
                      onClick={() =>
                        void handleMoveFavorites(
                          Array.from(selectedIds),
                          bulkProjectId,
                        )
                      }
                    >
                      Move
                    </Button>
                    <Select
                      value={bulkCopyProjectId}
                      onValueChange={setBulkCopyProjectId}
                    >
                      <SelectTrigger
                        className="h-11 w-full text-sm sm:h-8 sm:w-44"
                        aria-label="Copy selected favorites to project"
                      >
                        <SelectValue placeholder="Copy to project" />
                      </SelectTrigger>
                      <SelectContent>
                        {projects.map((project) => (
                          <SelectItem key={project.id} value={project.id}>
                            {project.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="h-11 sm:h-8"
                      disabled={!bulkCopyProjectId}
                      onClick={() =>
                        void handleCopyFavorites(
                          Array.from(selectedIds),
                          bulkCopyProjectId,
                        )
                      }
                    >
                      Copy
                    </Button>
                    <BulkTagActions
                      disabled={selectedIds.size === 0}
                      onSubmit={handleBulkUpdateTags}
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-11 sm:h-8"
                      onClick={() => setSelectedIds(new Set())}
                    >
                      Clear
                    </Button>
                  </div>
                </div>
              ) : null}

              {/* Shared panel for all four filter tabs; the triggers point
                  their aria-controls here (there are no per-tab panels). */}
              <div
                id="favorites-items-panel"
                role="tabpanel"
                aria-label="Saved items"
              >
              {loading ? (
                <div className="space-y-2" aria-label="Loading favorites">
                  {Array.from({ length: 4 }, (_, index) => (
                    <Skeleton key={index} className="h-28 w-full rounded-lg" />
                  ))}
                </div>
              ) : visible.length === 0 ? (
                <Card className="flex flex-col items-center gap-2 p-8 text-center">
                  <Star className="h-6 w-6 text-muted-foreground" />
                  <div className="text-sm font-medium">
                    {favorites.length === 0
                      ? "No favorites yet"
                      : "No favorites match these filters"}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {favorites.length === 0
                      ? "Click the star on any search result or agreement to save it here."
                      : "Try clearing some filters."}
                  </div>
                </Card>
              ) : (
                <div className="space-y-2">
                  {visible.map((fav) => (
                    <FavoriteRow
                      key={fav.id}
                      fav={fav}
                      agreement={
                        fav.agreement_uuid
                          ? (agreementByUuid[fav.agreement_uuid] ?? null)
                          : null
                      }
                      sectionDetails={
                        fav.item_type === "section"
                          ? sectionByUuid[fav.item_uuid]
                          : null
                      }
                      clauseTypeLabelById={clauseTypeLabelById}
                      selected={selectedIds.has(fav.id)}
                      onSelectChange={handleSelectFavorite}
                      onUpdated={handleUpdated}
                      onRemoved={handleRemoved}
                      onTagsChanged={handleTagsChanged}
                      onTagDeleted={handleTagDeleted}
                    />
                  ))}
                </div>
              )}
              </div>
            </div>
          </div>
        </div>
      </DndContext>
    </PageShell>
  );
}
