import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import jsxA11y from "eslint-plugin-jsx-a11y";
import globals from "globals";

export default tseslint.config(
  {
    ignores: [
      "dist/**",
      "node_modules/**",
      "shared/csp-script-hashes.generated.json",
    ],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node,
      },
    },
    rules: {
      // tsc (strict) already flags unused locals; ESLint re-flagging
      // intentional `_`-prefixed placeholders is just noise.
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrors: "none",
        },
      ],
    },
  },
  {
    files: ["**/*.{ts,tsx}"],
    extends: [reactHooks.configs.flat.recommended],
    rules: {
      // React Compiler-derived advisories. Real cleanups, but each one needs
      // a deliberate refactor of effect/memo structure — burn down over time
      // rather than gating CI on a rewrite of ~35 legacy sites.
      "react-hooks/set-state-in-effect": "warn",
      "react-hooks/preserve-manual-memoization": "warn",
    },
  },
  {
    files: ["**/*.tsx"],
    extends: [jsxA11y.flatConfigs.recommended],
    rules: {
      // Deliberate UX in dialogs and inline-rename inputs; WCAG permits
      // autofocus where focus is moved in response to a user action.
      "jsx-a11y/no-autofocus": "off",
      // role="list" on ul/ol is intentional: Safari/VoiceOver strip list
      // semantics when list-style is reset (Tailwind preflight does this).
      "jsx-a11y/no-redundant-roles": [
        "error",
        { ul: ["list"], ol: ["list"] },
      ],
      // Recognize our form primitives as controls so <label> wrapping an
      // <Input> isn't flagged.
      "jsx-a11y/label-has-associated-control": [
        "error",
        { controlComponents: ["Input", "Textarea", "Checkbox", "Select"] },
      ],
      // Focusable scrollable containers (<pre>, role="region") are the
      // WCAG-recommended way to make overflow content keyboard-reachable.
      "jsx-a11y/no-noninteractive-tabindex": [
        "error",
        { tags: ["pre"], roles: ["tabpanel", "region"] },
      ],
    },
  },
  {
    // Vendored shadcn/ui primitives: heading/anchor content arrives via
    // spread props the a11y rules can't see, and the sidebar skeleton uses
    // Math.random() for widths by design.
    files: ["client/components/ui/**"],
    rules: {
      "jsx-a11y/heading-has-content": "off",
      "jsx-a11y/anchor-has-content": "off",
      "react-hooks/purity": "warn",
    },
  },
  {
    // Ambient declaration files declare names for the compiler, not for use.
    files: ["**/*.d.ts"],
    rules: {
      "@typescript-eslint/no-unused-vars": "off",
    },
  },
);
