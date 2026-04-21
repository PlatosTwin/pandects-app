/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DISABLE_ANALYTICS?: string;
  readonly VITE_GA_DEBUG_MODE?: string;
  readonly VITE_GA_MEASUREMENT_ID?: string;
}

declare global {
  interface Window {
    dataLayer: unknown[];
    gtag: (...args: unknown[]) => void;
  }
}

declare module "*.md?raw" {
  const content: string;
  export default content;
}

export {};
