/**
 * Environment detection utility for conditional feature flags
 */

/**
 * Detects if the application is running in local development environment
 * @returns {boolean} true if running locally, false if in production
 */
export function isLocalEnvironment(): boolean {
  // Use Vite's built-in environment variable
  // PROD is true when built for production, false during development
  return !import.meta.env.PROD;
}

/**
 * Detects if the application is running in production environment
 * @returns {boolean} true if in production, false if running locally
 */
export function isProductionEnvironment(): boolean {
  return import.meta.env.PROD;
}
