import { Monitor, Moon, Sun } from "lucide-react";
import { memo, useEffect, useSyncExternalStore } from "react";
import { Button } from "@/components/ui/button";
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

function ThemeToggleComponent() {
  const preference = useSyncExternalStore(
    subscribeTheme,
    getThemePreference,
    getServerSnapshot,
  );

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
