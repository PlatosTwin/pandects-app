import { useEffect, useRef } from "react";

type Point = { x: number; y: number };
type Rect = { left: number; right: number; top: number; bottom: number };

type BallState = {
  position: Point;
  velocity: Point;
  angleRad: number;
  angularVelocityRadPerSec: number;
  lastTimestampMs: number | null;
  settledMs: number;
};

type Shard = {
  element: HTMLDivElement;
  position: Point;
  velocity: Point;
  angleRad: number;
  angularVelocityRadPerSec: number;
  lifeMs: number;
  maxLifeMs: number;
  sizePx: number;
};

type EndStyle = "shatter" | "fade";

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tagName = target.tagName.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select";
}

function prefersReducedMotion(): boolean {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function toLocalRect(containerRect: DOMRect, rect: DOMRect): Rect {
  return {
    left: rect.left - containerRect.left,
    right: rect.right - containerRect.left,
    top: rect.top - containerRect.top,
    bottom: rect.bottom - containerRect.top,
  };
}

function queryTargetRects(container: HTMLElement, targets: string[]): Rect[] {
  const containerRect = container.getBoundingClientRect();
  return targets
    .map((target) => {
      const element = container.querySelector<HTMLElement>(
        `[data-panda-target="${target}"]`,
      );
      if (!element) return null;
      const rect = element.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return null;
      return toLocalRect(containerRect, rect);
    })
    .filter((rect): rect is Rect => rect !== null);
}

function rand(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function almostZero(value: number, epsilon: number): boolean {
  return Math.abs(value) <= epsilon;
}

function normalizeAngle(angleRad: number): number {
  const twoPi = Math.PI * 2;
  const normalized = angleRad % twoPi;
  return normalized < 0 ? normalized + twoPi : normalized;
}

function parseEndStyle(value: unknown): EndStyle {
  return value === "fade" ? "fade" : "shatter";
}

export default function PandaEasterEgg({
  containerRef,
}: {
  containerRef: React.RefObject<HTMLElement | null>;
}) {
  const ballRef = useRef<HTMLDivElement | null>(null);
  const shardLayerRef = useRef<HTMLDivElement | null>(null);
  const isActiveRef = useRef(false);
  const phaseRef = useRef<"idle" | "ball" | "shatter">("idle");
  const rafIdRef = useRef<number | null>(null);
  const matchIndexRef = useRef(0);
  const matchTimerRef = useRef<number | null>(null);
  const ballStateRef = useRef<BallState | null>(null);
  const shardsRef = useRef<Shard[]>([]);
  const platformsRef = useRef<Rect[]>([]);
  const lastShatterTimestampMsRef = useRef<number | null>(null);

  useEffect(() => {
    const disabled = import.meta.env.VITE_DISABLE_PANDA_EASTER_EGG === "1";
    if (disabled) return;

    const container = containerRef.current;
    const ball = ballRef.current;
    const shardLayer = shardLayerRef.current;
    if (!container || !ball) return;
    if (!shardLayer) return;
    if (prefersReducedMotion()) return;

    const endStyle = parseEndStyle(import.meta.env.VITE_PANDA_END_STYLE);

    const resetMatch = () => {
      matchIndexRef.current = 0;
      if (matchTimerRef.current !== null) window.clearTimeout(matchTimerRef.current);
      matchTimerRef.current = null;
    };

    const clearShards = () => {
      for (const shard of shardsRef.current) {
        shard.element.remove();
      }
      shardsRef.current = [];
      lastShatterTimestampMsRef.current = null;
    };

    const stop = () => {
      isActiveRef.current = false;
      phaseRef.current = "idle";
      if (rafIdRef.current !== null) cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
      ballStateRef.current = null;
      clearShards();
      ball.style.opacity = "0";
      window.setTimeout(() => {
        if (!isActiveRef.current) ball.style.display = "none";
      }, 220);
    };

    const BALL_PHYSICS = {
      gravityPxPerSec2: 1500,
      airDragPerSec: 0.9,
      restitution: 0.55,
      groundFriction: 0.86,
      platformFriction: 0.9,
      restVYThreshold: 140,
      restVThreshold: 34,
      restAngularThreshold: 0.9,
      supportEpsilonPx: 3,
      supportedFrictionPerSec: 5.5,
      supportedSpinDampingPerSec: 2.2,
    } as const;

    const SHATTER_PHYSICS = {
      gravityPxPerSec2: 1700,
      dragPerSec: 1.5,
      restitution: 0.28,
      groundFriction: 0.7,
      groundPaddingPx: 6,
    } as const;

    const END_TIMING = {
      settleBeforeEndMs: 900,
      fadeDurationMs: 700,
    } as const;

    const initScene = (): {
      ballSizePx: number;
      initial: BallState;
      groundY: number;
      bounds: { width: number; height: number };
    } | null => {
      const containerRect = container.getBoundingClientRect();
      const logo = container.querySelector<HTMLElement>(
        `[data-panda-target="logo"]`,
      );
      if (!logo) return null;
      const logoRect = logo.getBoundingClientRect();
      if (logoRect.width <= 0 || logoRect.height <= 0) return null;

      const ballSizePx = Math.max(6, Math.round(logoRect.width / 4));

      const bounds = {
        width: Math.round(containerRect.width),
        height: Math.round(containerRect.height),
      };

      const groundY = bounds.height - ballSizePx - 6;

      const startX =
        logoRect.left -
        containerRect.left +
        logoRect.width * 0.48 -
        ballSizePx / 2;
      const startY =
        logoRect.top -
        containerRect.top +
        logoRect.height * 0.6 -
        ballSizePx / 2;

      platformsRef.current = queryTargetRects(container, [
        "brand",
        "nav-search",
        "nav-docs",
        "nav-bulk-data",
        "nav-about",
      ]);

      const initial: BallState = {
        position: {
          x: clamp(startX, 0, Math.max(0, bounds.width - ballSizePx)),
          y: clamp(startY, 0, Math.max(0, bounds.height - ballSizePx)),
        },
        velocity: {
          x: rand(520, 720),
          y: rand(-520, -380),
        },
        angleRad: 0,
        angularVelocityRadPerSec: rand(-6, 6),
        lastTimestampMs: null,
        settledMs: 0,
      };

      return { ballSizePx, initial, groundY, bounds };
    };

    const renderBall = (state: BallState, ballSizePx: number) => {
      ball.style.width = `${ballSizePx}px`;
      ball.style.height = `${ballSizePx}px`;
      ball.style.transform = `translate3d(${state.position.x}px, ${state.position.y}px, 0) rotate(${state.angleRad}rad)`;
    };

    const explodeToShards = (
      origin: Point,
      ballSizePx: number,
      platforms: Rect[],
      bounds: { width: number; height: number },
    ) => {
      clearShards();
      phaseRef.current = "shatter";

      const center = {
        x: origin.x + ballSizePx / 2,
        y: origin.y + ballSizePx / 2,
      };

      const shardCount = Math.round(rand(16, 26));
      const maxLifeMs = Math.round(rand(650, 950));

      for (let i = 0; i < shardCount; i += 1) {
        const sizePx = Math.round(rand(3, Math.max(4, ballSizePx * 0.35)));
        const element = document.createElement("div");
        element.style.position = "absolute";
        element.style.left = "0";
        element.style.top = "0";
        element.style.width = `${sizePx}px`;
        element.style.height = `${sizePx}px`;
        element.style.background = "black";
        element.style.borderRadius = `${Math.round(rand(0, 2))}px`;
        element.style.opacity = "1";
        element.style.willChange = "transform, opacity";
        element.style.pointerEvents = "none";

        const theta = rand(-Math.PI * 0.25, Math.PI * 0.25);
        const speed = rand(260, 680);
        const velocity = {
          x: Math.cos(theta) * speed,
          y: -Math.abs(Math.sin(theta)) * rand(220, 520),
        };

        const shard: Shard = {
          element,
          position: { x: center.x - sizePx / 2, y: center.y - sizePx / 2 },
          velocity,
          angleRad: rand(0, Math.PI * 2),
          angularVelocityRadPerSec: rand(-18, 18),
          lifeMs: 0,
          maxLifeMs,
          sizePx,
        };

        shardsRef.current.push(shard);
        shardLayer.appendChild(element);
      }

      ball.style.display = "none";
      lastShatterTimestampMsRef.current = null;

      const stepShatter = (timestampMs: number) => {
        if (!isActiveRef.current || phaseRef.current !== "shatter") return;

        const last = lastShatterTimestampMsRef.current ?? timestampMs;
        lastShatterTimestampMsRef.current = timestampMs;
        const dt = clamp((timestampMs - last) / 1000, 0, 1 / 30);

        const {
          gravityPxPerSec2,
          dragPerSec,
          restitution,
          groundFriction,
          groundPaddingPx,
        } = SHATTER_PHYSICS;

        const groundY = bounds.height - groundPaddingPx;

        let aliveCount = 0;
        for (const shard of shardsRef.current) {
          const previous = { ...shard.position };
          shard.lifeMs += dt * 1000;
          if (shard.lifeMs >= shard.maxLifeMs) {
            shard.element.style.opacity = "0";
            continue;
          }
          aliveCount += 1;

          shard.velocity.y += gravityPxPerSec2 * dt;
          shard.velocity.x *= Math.exp(-dragPerSec * dt);
          shard.velocity.y *= Math.exp(-dragPerSec * dt * 0.3);

          shard.position.x += shard.velocity.x * dt;
          shard.position.y += shard.velocity.y * dt;
          shard.angleRad = normalizeAngle(
            shard.angleRad + shard.angularVelocityRadPerSec * dt,
          );
          shard.angularVelocityRadPerSec *= Math.exp(-dt * 0.8);

          const bottom = shard.position.y + shard.sizePx;
          if (bottom >= groundY) {
            shard.position.y = groundY - shard.sizePx;
            if (shard.velocity.y > 0) shard.velocity.y = -shard.velocity.y * restitution;
            shard.velocity.x *= groundFriction;
          }

          for (const platform of platforms) {
            const platformTop = platform.top;
            const prevBottom = previous.y + shard.sizePx;
            const nextBottom = shard.position.y + shard.sizePx;
            const overlapsX =
              shard.position.x + shard.sizePx >= platform.left - 4 &&
              shard.position.x <= platform.right + 4;
            const crossedTop = prevBottom < platformTop && nextBottom >= platformTop;
            if (crossedTop && overlapsX && shard.velocity.y > 0) {
              shard.position.y = platformTop - shard.sizePx;
              shard.velocity.y = -shard.velocity.y * restitution;
              shard.velocity.x *= 0.86;
              break;
            }
          }

          const t = shard.lifeMs / shard.maxLifeMs;
          shard.element.style.opacity = `${1 - t}`;
          shard.element.style.transform = `translate3d(${shard.position.x}px, ${shard.position.y}px, 0) rotate(${shard.angleRad}rad)`;
        }

        if (aliveCount === 0) {
          stop();
          return;
        }

        rafIdRef.current = requestAnimationFrame(stepShatter);
      };

      rafIdRef.current = requestAnimationFrame(stepShatter);
    };

    const stepBall = (
      timestampMs: number,
      config: {
        ballSizePx: number;
        groundY: number;
        bounds: { width: number; height: number };
      },
    ) => {
      const ballSizePx = config.ballSizePx;

      const state = ballStateRef.current;
      if (!state || !isActiveRef.current || phaseRef.current !== "ball") return;

      const last = state.lastTimestampMs ?? timestampMs;
      state.lastTimestampMs = timestampMs;
      const dt = clamp((timestampMs - last) / 1000, 0, 1 / 30);

      const {
        gravityPxPerSec2,
        airDragPerSec,
        restitution,
        groundFriction,
        platformFriction,
        restVYThreshold,
        restVThreshold,
        restAngularThreshold,
        supportEpsilonPx,
        supportedFrictionPerSec,
        supportedSpinDampingPerSec,
      } = BALL_PHYSICS;

      const previous = { ...state.position };

      state.velocity.y += gravityPxPerSec2 * dt;
      state.velocity.x *= Math.exp(-airDragPerSec * dt * 0.15);
      state.velocity.y *= Math.exp(-airDragPerSec * dt * 0.02);

      state.position.x += state.velocity.x * dt;
      state.position.y += state.velocity.y * dt;

      state.angleRad = normalizeAngle(
        state.angleRad + state.angularVelocityRadPerSec * dt,
      );
      state.angularVelocityRadPerSec *= Math.exp(-dt * 0.6);

      const maxX = Math.max(0, config.bounds.width - ballSizePx);
      const maxY = Math.max(0, config.bounds.height - ballSizePx);

      if (state.position.x < 0) {
        state.position.x = 0;
        state.velocity.x = Math.abs(state.velocity.x) * restitution;
        state.angularVelocityRadPerSec += rand(-2, 2);
      } else if (state.position.x > maxX) {
        state.position.x = maxX;
        state.velocity.x = -Math.abs(state.velocity.x) * restitution;
        state.angularVelocityRadPerSec += rand(-2, 2);
      }

      if (state.position.y < 0) {
        state.position.y = 0;
        if (state.velocity.y < 0) {
          state.velocity.y = Math.abs(state.velocity.y) * restitution;
          state.velocity.x *= 0.96;
          state.angularVelocityRadPerSec += rand(-2, 2);
        }
      } else if (state.position.y > maxY) {
        state.position.y = maxY;
      }

      const wasFalling = state.velocity.y > 0;
      const hitGround =
        previous.y < config.groundY - supportEpsilonPx &&
        state.position.y >= config.groundY;

      if (state.position.y >= config.groundY) {
        state.position.y = config.groundY;
        if (wasFalling) {
          const impactVY = Math.abs(state.velocity.y);
          if (hitGround && impactVY >= restVYThreshold) {
            state.velocity.y = -impactVY * restitution;
            state.velocity.x *= groundFriction;
            state.angularVelocityRadPerSec =
              state.angularVelocityRadPerSec * 0.7 + rand(-2, 2);
          } else {
            state.velocity.y = 0;
            state.velocity.x *= groundFriction;
            state.angularVelocityRadPerSec *= 0.65;
          }
        } else if (state.velocity.y > 0 && Math.abs(state.velocity.y) < restVYThreshold) {
          state.velocity.y = 0;
        }
      }

      if (state.velocity.y > 0) {
        const platforms = platformsRef.current;
        for (const platform of platforms) {
          const platformTop = platform.top;
          const platformLeft = platform.left - 6;
          const platformRight = platform.right + 6;

          const prevBottom = previous.y + ballSizePx;
          const nextBottom = state.position.y + ballSizePx;
          const crossedTop =
            prevBottom < platformTop - supportEpsilonPx && nextBottom >= platformTop;
          const overlapsX =
            state.position.x + ballSizePx >= platformLeft &&
            state.position.x <= platformRight;

          if (crossedTop && overlapsX) {
            state.position.y = platformTop - ballSizePx;
            const impactVY = Math.abs(state.velocity.y);
            if (impactVY >= restVYThreshold) {
              state.velocity.y = -impactVY * restitution;
              state.velocity.x *= platformFriction;
              state.angularVelocityRadPerSec =
                state.angularVelocityRadPerSec * 0.72 + rand(-2.5, 2.5);
            } else {
              state.velocity.y = 0;
              state.velocity.x *= platformFriction;
              state.angularVelocityRadPerSec *= 0.68;
            }
            break;
          }
        }
      }

      let isSupported = false;
      if (Math.abs(state.position.y - config.groundY) <= supportEpsilonPx) {
        isSupported = true;
        state.position.y = config.groundY;
        if (state.velocity.y > 0 && Math.abs(state.velocity.y) < restVYThreshold) {
          state.velocity.y = 0;
        }
      } else {
        const platforms = platformsRef.current;
        for (const platform of platforms) {
          const platformTop = platform.top;
          const platformLeft = platform.left - 6;
          const platformRight = platform.right + 6;
          const bottomY = state.position.y + ballSizePx;

          const nearTop = Math.abs(bottomY - platformTop) <= supportEpsilonPx;
          const overlapsX =
            state.position.x + ballSizePx >= platformLeft &&
            state.position.x <= platformRight;

          if (nearTop && overlapsX) {
            isSupported = true;
            state.position.y = platformTop - ballSizePx;
            if (state.velocity.y > 0 && Math.abs(state.velocity.y) < restVYThreshold) {
              state.velocity.y = 0;
            }
            break;
          }
        }
      }

      if (isSupported) {
        state.velocity.x *= Math.exp(-supportedFrictionPerSec * dt);
        if (almostZero(state.velocity.x, 10)) state.velocity.x = 0;
        state.angularVelocityRadPerSec *= Math.exp(-dt * supportedSpinDampingPerSec);
        if (almostZero(state.angularVelocityRadPerSec, 0.25)) {
          state.angularVelocityRadPerSec = 0;
        }
      }

      const linearSpeed = Math.hypot(state.velocity.x, state.velocity.y);
      if (
        isSupported &&
        linearSpeed < restVThreshold &&
        Math.abs(state.angularVelocityRadPerSec) < restAngularThreshold
      ) {
        state.settledMs += dt * 1000;
      } else {
        state.settledMs = 0;
      }

      const fadeStartMs = END_TIMING.settleBeforeEndMs;
      const fadeDurationMs = END_TIMING.fadeDurationMs;
      if (endStyle === "fade") {
        const fadeT = clamp(
          (state.settledMs - fadeStartMs) / fadeDurationMs,
          0,
          1,
        );
        ball.style.opacity = `${1 - fadeT}`;
      } else {
        ball.style.opacity = "1";
        if (state.settledMs > fadeStartMs) {
          explodeToShards(
            state.position,
            ballSizePx,
            platformsRef.current,
            config.bounds,
          );
          return;
        }
      }

      renderBall(state, ballSizePx);

      if (state.settledMs > fadeStartMs + fadeDurationMs + 80) {
        stop();
        return;
      }

      rafIdRef.current = requestAnimationFrame((t) => stepBall(t, config));
    };

    const toggle = () => {
      if (isActiveRef.current) {
        stop();
        return;
      }

      const scene = initScene();
      if (!scene) return;

      isActiveRef.current = true;
      phaseRef.current = "ball";
      ballStateRef.current = scene.initial;

      ball.style.display = "block";
      ball.style.opacity = "1";
      ball.style.transition = "opacity 220ms ease-out";

      renderBall(scene.initial, scene.ballSizePx);
      rafIdRef.current = requestAnimationFrame((t) =>
        stepBall(t, {
          ballSizePx: scene.ballSizePx,
          groundY: scene.groundY,
          bounds: scene.bounds,
        }),
      );
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (isEditableTarget(event.target)) return;
      if (event.repeat) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      const key = event.key.toLowerCase();
      if (key.length !== 1 || key < "a" || key > "z") return;

      const secret = "panda";
      const nextExpected = secret[matchIndexRef.current];

      if (key === nextExpected) {
        matchIndexRef.current += 1;
        if (matchTimerRef.current !== null) window.clearTimeout(matchTimerRef.current);
        matchTimerRef.current = window.setTimeout(resetMatch, 1000);
      } else if (key === secret[0]) {
        matchIndexRef.current = 1;
        if (matchTimerRef.current !== null) window.clearTimeout(matchTimerRef.current);
        matchTimerRef.current = window.setTimeout(resetMatch, 1000);
      } else {
        resetMatch();
      }

      if (matchIndexRef.current === secret.length) {
        resetMatch();
        toggle();
      }
    };

    window.addEventListener("keydown", onKeyDown, { capture: true });
    return () => {
      resetMatch();
      stop();
      window.removeEventListener("keydown", onKeyDown, { capture: true });
    };
  }, [containerRef]);

  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 z-0">
      <div ref={shardLayerRef} className="absolute inset-0" />
      <div
        ref={ballRef}
        className="absolute left-0 top-0 z-[1] hidden"
        style={{ willChange: "transform, opacity", opacity: 0 }}
      >
        <svg
          width="100%"
          height="100%"
          viewBox="0 0 100 100"
          role="presentation"
        >
          <circle cx="50" cy="50" r="48" fill="black" />
        </svg>
      </div>
    </div>
  );
}
