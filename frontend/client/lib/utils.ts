import { clsx, type ClassValue } from "clsx";
import { extendTailwindMerge } from "tailwind-merge";

// Teach tailwind-merge the elevation scale (tailwind.config.ts boxShadow), so
// `cn("shadow-raised", "shadow-overlay")` keeps only the later one.
const twMerge = extendTailwindMerge({
  extend: { classGroups: { shadow: [{ shadow: ["raised", "lifted", "overlay"] }] } },
});

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
