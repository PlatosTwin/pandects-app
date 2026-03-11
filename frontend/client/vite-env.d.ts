/// <reference types="vite/client" />

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
