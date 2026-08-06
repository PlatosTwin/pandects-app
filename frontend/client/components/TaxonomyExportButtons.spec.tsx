// @vitest-environment jsdom
/**
 * Behavioral contract for the taxonomy copy/download controls: the clipboard
 * receives the tree text, the download produces a .txt blob under the given
 * file name, and both are inert while there is nothing to export.
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { TaxonomyExportButtons } from "./TaxonomyExportButtons";

const TREE_TEXT = "Corporate [L1-CORP] (1 groups, 0 types)\n└── Group [L2-A] (0 types)";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function setClipboard(writeText: (value: string) => Promise<void>) {
  Object.defineProperty(navigator, "clipboard", {
    value: { writeText },
    configurable: true,
    writable: true,
  });
}

describe("TaxonomyExportButtons", () => {
  beforeEach(() => {
    setClipboard(vi.fn().mockResolvedValue(undefined));
  });

  it("writes the tree text to the clipboard and confirms the copy", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    setClipboard(writeText);

    render(
      <TaxonomyExportButtons
        getText={() => TREE_TEXT}
        fileName="taxonomy_2026-08-06.txt"
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /copy taxonomy tree/i }));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(TREE_TEXT));
    await screen.findByText("Copied");
    expect(screen.getByRole("status").textContent).toBe(
      "Taxonomy tree copied to clipboard",
    );
  });

  it("surfaces a failed clipboard write instead of claiming success", async () => {
    setClipboard(vi.fn().mockRejectedValue(new Error("denied")));

    render(
      <TaxonomyExportButtons
        getText={() => TREE_TEXT}
        fileName="taxonomy_2026-08-06.txt"
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /copy taxonomy tree/i }));

    await screen.findByText("Copy failed");
    expect(screen.queryByText("Copied")).toBeNull();
  });

  it("downloads a text/plain blob named after the taxonomy", async () => {
    const createObjectURL = vi.fn().mockReturnValue("blob:taxonomy");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL,
      revokeObjectURL,
    });

    const clicked: HTMLAnchorElement[] = [];
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function (this: HTMLAnchorElement) {
        clicked.push(this);
      });

    render(
      <TaxonomyExportButtons
        getText={() => TREE_TEXT}
        fileName="taxonomy_2026-08-06.txt"
      />,
    );

    fireEvent.click(
      screen.getByRole("button", { name: /download taxonomy tree/i }),
    );

    expect(clickSpy).toHaveBeenCalledTimes(1);
    expect(clicked[0].download).toBe("taxonomy_2026-08-06.txt");
    expect(clicked[0].getAttribute("href")).toBe("blob:taxonomy");
    expect(clicked[0].isConnected).toBe(false);

    const blob = createObjectURL.mock.calls[0][0] as Blob;
    expect(blob.type).toBe("text/plain;charset=utf-8;");
    await expect(blob.text()).resolves.toBe(TREE_TEXT);

    vi.unstubAllGlobals();
  });

  it("disables both controls when there is nothing to export", () => {
    const getText = vi.fn().mockReturnValue("");
    render(
      <TaxonomyExportButtons
        getText={getText}
        fileName="taxonomy_2026-08-06.txt"
        disabled
      />,
    );

    for (const name of [/copy taxonomy tree/i, /download taxonomy tree/i]) {
      const button = screen.getByRole("button", { name }) as HTMLButtonElement;
      expect(button.disabled).toBe(true);
      fireEvent.click(button);
    }
    expect(getText).not.toHaveBeenCalled();
  });
});
