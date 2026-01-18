/**
 * Production-safe logging utility
 * Only logs in development mode
 */

const isDev = import.meta.env.DEV;

export const logger = {
  error: (...args: unknown[]) => {
    if (isDev) {
      console.error(...args);
    }
  },
  warn: (...args: unknown[]) => {
    if (isDev) {
      console.warn(...args);
    }
  },
  log: (...args: unknown[]) => {
    if (isDev) {
      console.log(...args);
    }
  },
};
