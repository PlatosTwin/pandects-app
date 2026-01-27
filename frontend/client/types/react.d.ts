import "react";

declare module "react" {
  interface ImgHTMLAttributes<T> {
    // React doesn't type the lowercase HTML attribute, but we need it to avoid DOM prop warnings.
    fetchpriority?: "high" | "low" | "auto";
  }
}
