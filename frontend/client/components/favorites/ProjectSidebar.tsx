import { useDroppable } from "@dnd-kit/core";
import { useState } from "react";
import { Check, ChevronDown, Folder, Pencil, Plus, Trash2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { TagSwatch } from "@/components/favorites/TagPill";
import { useFavorites } from "@/contexts/FavoritesContext";
import { useToast } from "@/hooks/use-toast";
import {
  TAG_COLORS,
  type FavoriteProject,
  type TagColor,
} from "@/lib/favorites-api";

function ProjectDropButton({
  project,
  active,
  count,
  onSelect,
}: {
  project: FavoriteProject;
  active: boolean;
  count: number;
  onSelect: (projectId: string) => void;
}) {
  const { isOver, setNodeRef } = useDroppable({
    id: `project:${project.id}`,
    data: { projectId: project.id },
  });
  return (
    <button
      ref={setNodeRef}
      type="button"
      onClick={() => onSelect(project.id)}
      aria-pressed={active}
      className={`flex w-full items-center justify-between gap-2 rounded-md border px-2.5 py-2 text-left text-sm transition-colors ${
        active
          ? "border-primary bg-primary/10 text-foreground"
          : "border-transparent bg-background hover:border-border hover:bg-muted/60"
      } ${isOver ? "ring-2 ring-primary ring-offset-1 ring-offset-background" : ""}`}
    >
      <span className="flex min-w-0 items-center gap-2">
        <TagSwatch color={project.color} className="h-3 w-3 shrink-0" />
        <span className="truncate font-medium">{project.name}</span>
      </span>
      <span className="rounded-full bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
        {count}
      </span>
    </button>
  );
}

export function ProjectSidebar({
  projects,
  activeProjectId,
  projectCounts,
  onSelectProject,
  onProjectCreated,
  onProjectUpdated,
  onProjectDeleted,
}: {
  projects: FavoriteProject[];
  activeProjectId: string | null;
  projectCounts: Record<string, number>;
  onSelectProject: (projectId: string | null) => void;
  onProjectCreated: (project: FavoriteProject) => void;
  onProjectUpdated: (project: FavoriteProject) => void;
  onProjectDeleted: (projectId: string, reassignProjectId: string) => void;
}) {
  const { toast } = useToast();
  const { createProject, updateProject, removeProject } = useFavorites();
  const [newName, setNewName] = useState("");
  const [newColor, setNewColor] = useState<TagColor>("blue");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [draftColor, setDraftColor] = useState<TagColor>("blue");
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(true);

  const total = Object.values(projectCounts).reduce(
    (sum, count) => sum + count,
    0,
  );

  const beginEdit = (project: FavoriteProject) => {
    setEditingId(project.id);
    setDraftName(project.name);
    setDraftColor(project.color);
  };

  const cancelEdit = () => {
    setEditingId(null);
    setDraftName("");
    setDraftColor("blue");
  };

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    setBusy(true);
    try {
      const project = await createProject(name, newColor);
      onProjectCreated(project);
      setNewName("");
      setNewColor("blue");
    } catch {
      toast({ title: "Couldn't create project", variant: "destructive" });
    } finally {
      setBusy(false);
    }
  };

  const handleSave = async (project: FavoriteProject) => {
    const name = draftName.trim();
    if (!name) return;
    setBusy(true);
    try {
      const updated = await updateProject(project.id, {
        name,
        color: draftColor,
      });
      onProjectUpdated(updated);
      cancelEdit();
    } catch {
      toast({ title: "Couldn't update project", variant: "destructive" });
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (project: FavoriteProject) => {
    const target = projects.find((candidate) => candidate.id !== project.id);
    if (!target) {
      toast({
        title: "Create another project first",
        description:
          "Favorites need a destination before this project can be deleted.",
        variant: "destructive",
      });
      return;
    }
    const confirmed = window.confirm(
      `Delete "${project.name}" and move its favorites to "${target.name}"?`,
    );
    if (!confirmed) return;
    setBusy(true);
    try {
      const result = await removeProject(project.id, target.id);
      onProjectDeleted(project.id, result.reassigned_to_project_id);
      if (activeProjectId === project.id) {
        onSelectProject(result.reassigned_to_project_id);
      }
    } catch {
      toast({ title: "Couldn't delete project", variant: "destructive" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <aside className="self-start rounded-lg border bg-background p-3 shadow-sm lg:sticky lg:top-20">
      <Collapsible open={open} onOpenChange={setOpen}>
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            <Folder className="h-3.5 w-3.5" aria-hidden="true" />
            Projects
          </div>
          <div className="flex items-center gap-1">
            <Badge variant="secondary">{projects.length}</Badge>
            <CollapsibleTrigger asChild>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-11 w-11 sm:h-7 sm:w-7"
                aria-label={open ? "Collapse projects" : "Expand projects"}
              >
                <ChevronDown
                  className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`}
                  aria-hidden="true"
                />
              </Button>
            </CollapsibleTrigger>
          </div>
        </div>

        <CollapsibleContent className="mt-3 space-y-3">
          <div className="space-y-1.5">
            <button
              type="button"
              onClick={() => onSelectProject(null)}
              aria-pressed={activeProjectId === null}
              className={`flex w-full items-center justify-between gap-2 rounded-md border px-2.5 py-2 text-left text-sm transition-colors ${
                activeProjectId === null
                  ? "border-primary bg-primary/10 text-foreground"
                  : "border-transparent bg-background hover:border-border hover:bg-muted/60"
              }`}
            >
              <span className="font-medium">All projects</span>
              <span className="rounded-full bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                {total}
              </span>
            </button>
            {projects.map((project) => (
              <div key={project.id} className="rounded-md">
                {editingId === project.id ? (
                  <div className="space-y-2 rounded-md border p-2">
                    <Input
                      aria-label={`Project name for ${project.name}`}
                      value={draftName}
                      onChange={(e) => setDraftName(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !busy) {
                          e.preventDefault();
                          void handleSave(project);
                        }
                      }}
                      className="h-11 text-sm sm:h-8"
                      autoFocus
                    />
                    <div className="flex flex-wrap gap-1.5">
                      {TAG_COLORS.map((color) => (
                        <button
                          key={color}
                          type="button"
                          onClick={() => setDraftColor(color)}
                          aria-label={`Use ${color} color`}
                          aria-pressed={draftColor === color}
                          className={
                            "relative inline-grid h-11 w-11 place-items-center rounded-full ring-1 ring-transparent transition-shadow sm:h-6 sm:w-6 " +
                            (draftColor === color
                              ? "ring-2 ring-foreground ring-offset-2 ring-offset-background"
                              : "hover:ring-muted-foreground/40")
                          }
                        >
                          <TagSwatch color={color} className="h-4 w-4" />
                          {draftColor === color ? (
                            <Check className="pointer-events-none absolute left-1/2 top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 text-white drop-shadow" />
                          ) : null}
                        </button>
                      ))}
                    </div>
                    <div className="flex justify-end gap-1">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-11 px-3 text-xs sm:h-7 sm:px-2"
                        onClick={cancelEdit}
                        disabled={busy}
                      >
                        Cancel
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        className="h-11 px-3 text-xs sm:h-7 sm:px-2"
                        onClick={() => void handleSave(project)}
                        disabled={busy || !draftName.trim()}
                      >
                        Save
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-[1fr_auto] items-center gap-1">
                    <ProjectDropButton
                      project={project}
                      active={activeProjectId === project.id}
                      count={projectCounts[project.id] ?? 0}
                      onSelect={onSelectProject}
                    />
                    <div className="flex">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="h-11 w-11 sm:h-8 sm:w-8"
                        aria-label={`Edit project ${project.name}`}
                        title={`Edit project ${project.name}`}
                        onClick={() => beginEdit(project)}
                        disabled={busy}
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="h-11 w-11 text-muted-foreground hover:text-destructive sm:h-8 sm:w-8"
                        aria-label={`Delete project ${project.name}`}
                        title={`Delete project ${project.name}`}
                        onClick={() => void handleDelete(project)}
                        disabled={busy}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="space-y-2 border-t pt-3">
            <div className="text-xs font-medium text-muted-foreground">
              New project
            </div>
            <Input
              aria-label="New project name"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !busy) {
                  e.preventDefault();
                  void handleCreate();
                }
              }}
              placeholder="Project name"
              className="h-11 text-sm sm:h-8"
            />
            <div className="flex flex-wrap gap-1.5">
              {TAG_COLORS.map((color) => (
                <button
                  key={color}
                  type="button"
                  onClick={() => setNewColor(color)}
                  aria-label={`Use ${color} color`}
                  aria-pressed={newColor === color}
                  className={
                    "relative inline-grid h-11 w-11 place-items-center rounded-full ring-1 ring-transparent transition-shadow sm:h-6 sm:w-6 " +
                    (newColor === color
                      ? "ring-2 ring-foreground ring-offset-2 ring-offset-background"
                      : "hover:ring-muted-foreground/40")
                  }
                >
                  <TagSwatch color={color} className="h-4 w-4" />
                  {newColor === color ? (
                    <Check className="pointer-events-none absolute left-1/2 top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 text-white drop-shadow" />
                  ) : null}
                </button>
              ))}
            </div>
            <Button
              type="button"
              size="sm"
              className="h-11 w-full gap-1.5 sm:h-8"
              onClick={() => void handleCreate()}
              disabled={busy || !newName.trim()}
            >
              <Plus className="h-3.5 w-3.5" aria-hidden="true" />
              Create
            </Button>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </aside>
  );
}
