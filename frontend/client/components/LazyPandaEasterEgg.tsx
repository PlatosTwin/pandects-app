import type { RefObject } from "react";
import { useEffect, useState } from "react";

type LazyPandaProps = {
  containerRef: RefObject<HTMLElement | null>;
};

type PandaComponent = React.ComponentType<LazyPandaProps>;

export function LazyPandaEasterEgg({ containerRef }: LazyPandaProps) {
  const [Component, setComponent] = useState<PandaComponent | null>(null);

  useEffect(() => {
    if (import.meta.env.VITE_DISABLE_PANDA_EASTER_EGG === "1") return;
    let cancelled = false;

    const load = () => {
      void import("@/components/PandaEasterEgg").then((mod) => {
        if (!cancelled) {
          setComponent(() => mod.default);
        }
      });
    };

    const handle = window.requestIdleCallback
      ? window.requestIdleCallback(load, { timeout: 1500 })
      : window.setTimeout(load, 1200);

    return () => {
      cancelled = true;
      if (window.cancelIdleCallback) {
        window.cancelIdleCallback(handle as number);
      } else {
        window.clearTimeout(handle as number);
      }
    };
  }, []);

  if (!Component) return null;
  return <Component containerRef={containerRef} />;
}
