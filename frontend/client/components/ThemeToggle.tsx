import { Monitor, Moon, Sun } from "lucide-react";
import { memo, useEffect, useSyncExternalStore } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  applyTheme,
  getThemePreference,
  setThemePreference,
  subscribeTheme,
  type ThemePreference,
} from "@/lib/theme";

const THEME_OPTIONS = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
] as const satisfies ReadonlyArray<{
  value: ThemePreference;
  label: string;
  icon: typeof Sun;
}>;

function getServerSnapshot(): ThemePreference {
  return "system";
}

function useThemePreference() {
  return useSyncExternalStore(subscribeTheme, getThemePreference, getServerSnapshot);
}

function ThemeToggleComponent() {
  const preference = useThemePreference();

  // The pre-paint script only toggles the class; re-apply once on mount to
  // sync the derived outputs it skips (meta theme-color).
  useEffect(() => {
    applyTheme();
  }, []);

  const current =
    THEME_OPTIONS.find((option) => option.value === preference) ??
    THEME_OPTIONS[2];
  const CurrentIcon = current.icon;

  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-9 w-9 rounded-md text-muted-foreground hover:text-foreground"
          aria-label={`Theme: ${current.label}`}
        >
          <CurrentIcon className="h-4 w-4" aria-hidden="true" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-36">
        <DropdownMenuRadioGroup
          value={preference}
          onValueChange={(value) =>
            setThemePreference(value as ThemePreference)
          }
        >
          {THEME_OPTIONS.map(({ value, label, icon: Icon }) => (
            <DropdownMenuRadioItem key={value} value={value}>
              <Icon
                className="mr-2 h-4 w-4 text-muted-foreground"
                aria-hidden="true"
              />
              {label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export const ThemeToggle = memo(ThemeToggleComponent);

// Inline segmented variant for the mobile menu sheet, where a nested dropdown
// is awkward to reach. Arrow keys are not wired: each option is a plain
// toggle button reachable with Tab.
function ThemeToggleInlineComponent({ className }: { className?: string }) {
  const preference = useThemePreference();

  return (
    <div
      role="group"
      aria-label="Theme"
      className={cn(
        "inline-flex items-center gap-0.5 rounded-md border border-border bg-muted/40 p-0.5",
        className,
      )}
    >
      {THEME_OPTIONS.map(({ value, label, icon: Icon }) => {
        const selected = preference === value;
        return (
          <button
            key={value}
            type="button"
            aria-pressed={selected}
            aria-label={label}
            title={label}
            onClick={() => setThemePreference(value)}
            className={cn(
              "inline-flex h-8 w-9 items-center justify-center rounded-[5px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              selected
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <Icon className="h-4 w-4" aria-hidden="true" />
          </button>
        );
      })}
    </div>
  );
}

export const ThemeToggleInline = memo(ThemeToggleInlineComponent);
