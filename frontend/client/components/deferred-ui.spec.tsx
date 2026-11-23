// @vitest-environment jsdom
/**
 * The BETA notice and the toaster load in their own chunks, after the entry.
 * These tests pin the behavior that must survive that split: the notice opens
 * on the first click, and a toast raised before the toaster loads still shows.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AppProviders } from "@/components/AppProviders";
import { NavBetaBadge } from "@/components/NavBetaBadge";
import { toast } from "@/hooks/use-toast";
import { createQueryClient } from "@/lib/query-client";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("NavBetaBadge", () => {
  it("opens the beta notice on the first click and closes it", async () => {
    render(<NavBetaBadge />);
    expect(screen.queryByRole("alertdialog")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Read beta notice" }));
    const dialog = await screen.findByRole("alertdialog");
    expect(dialog.textContent).toContain("Beta Notice");

    fireEvent.click(screen.getByRole("button", { name: "Got it" }));
    await vi.waitFor(() => {
      expect(screen.queryByRole("alertdialog")).toBeNull();
    });
    // The dialog has no Radix trigger; focus must still return to the badge.
    await vi.waitFor(() => {
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "Read beta notice" }),
      );
    });
  });
});

function RaiseToastOnMount() {
  useEffect(() => {
    toast({ title: "Saved before the toaster loaded" });
  }, []);
  return null;
}

describe("AppProviders toaster", () => {
  it("shows a toast raised before the deferred toaster loads", async () => {
    // The auth check has no backend here; fail it fast.
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new Error("offline"))),
    );
    render(
      <AppProviders queryClient={createQueryClient()}>
        <RaiseToastOnMount />
      </AppProviders>,
    );
    expect(
      await screen.findByText("Saved before the toaster loaded"),
    ).toBeTruthy();
  });
});
