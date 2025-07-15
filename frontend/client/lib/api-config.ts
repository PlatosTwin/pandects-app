/**
 * API configuration utility for environment-based API base URLs
 */

function getApiBaseUrl(): string {
  // Check if we're in production (build environment)
  const isProduction = import.meta.env.PROD;

  if (isProduction) {
    // Production environment (fly.io)
    return "https://pandects-api.fly.dev";
  } else {
    // Development environment (local)
    return "http://127.0.0.1:8080";
  }
}

export const API_BASE_URL = getApiBaseUrl();

/**
 * Helper function to construct API endpoint URLs
 */
export function apiUrl(endpoint: string): string {
  // Remove leading slash if present to avoid double slashes
  const cleanEndpoint = endpoint.startsWith("/") ? endpoint.slice(1) : endpoint;
  return `${API_BASE_URL}/${cleanEndpoint}`;
}
