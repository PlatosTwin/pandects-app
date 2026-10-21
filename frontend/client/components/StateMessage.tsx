import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { AlertTriangle, Inbox, RotateCw } from "lucide-react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type IconType = LucideIcon;

/**
 * Structural error shape. Matches the `ApiError` class (status/code/message)
 * without importing it, and also accepts plain `Error`s and strings.
 */
export type ErrorLike =
  | {
      status?: number;
      code?: string;
      message?: string;
      name?: string;
      serverMessage?: string;
    }
  | string
  | null
  | undefined;

export interface DescribedError {
  title: string;
  body: string;
  /** True when retrying the same request may succeed. */
  retryable: boolean;
  /** True when signing in would unlock the content. */
  needsSignIn: boolean;
}

/** Maps an error to user-facing copy. Never shows raw "Failed to fetch". */
export function describeError(error: ErrorLike): DescribedError {
  const shape = typeof error === "string" ? { message: error } : (error ?? {});
  const status = shape.status;
  const message = shape.message ?? "";
  const isNetwork =
    status === 0 ||
    shape.code === "network" ||
    shape.code === "network_error" ||
    shape.name === "TypeError" ||
    /failed to fetch|networkerror|load failed|network request failed/i.test(
      message,
    );

  // A 403 with its own explanation (lockdown, unverified email) isn't a sign-in problem.
  if (
    status === 403 &&
    shape.serverMessage &&
    !/sign(ed)?[- ]in|authentication required/i.test(shape.serverMessage)
  ) {
    return {
      title: "You don't have access to this",
      body: shape.serverMessage,
      retryable: false,
      needsSignIn: false,
    };
  }
  if (status === 401 || status === 403) {
    return {
      title: "Sign in to view this",
      body: "This content is available to signed-in accounts.",
      retryable: false,
      needsSignIn: true,
    };
  }
  if (status === 404) {
    return {
      title: "Not found",
      body: "We couldn't find what you were looking for.",
      retryable: false,
      needsSignIn: false,
    };
  }
  if (status === 429) {
    return {
      title: "Too many requests",
      body: "You're going a little fast. Wait a moment, then try again.",
      retryable: true,
      needsSignIn: false,
    };
  }
  if (isNetwork) {
    return {
      title: "Couldn't reach the server",
      body: "Check your connection, then try again.",
      retryable: true,
      needsSignIn: false,
    };
  }
  if (status !== undefined && status >= 500) {
    return {
      title: "Something went wrong on our end",
      body: "The server had a problem with this request. Try again in a moment.",
      retryable: true,
      needsSignIn: false,
    };
  }
  return {
    title: "Something went wrong",
    body: "This section couldn't be loaded. Try again in a moment.",
    retryable: true,
    needsSignIn: false,
  };
}

interface StateFrameProps {
  icon: IconType;
  iconClassName?: string;
  title: ReactNode;
  body?: ReactNode;
  action?: ReactNode;
  role?: "alert" | "status";
  /** `inline` for inside a card/panel; `card` draws its own bordered box. */
  variant?: "inline" | "card";
  className?: string;
}

function StateFrame({
  icon: Icon,
  iconClassName,
  title,
  body,
  action,
  role,
  variant = "card",
  className,
}: StateFrameProps) {
  return (
    <div
      role={role}
      className={cn(
        "flex flex-col items-center justify-center px-6 py-10 text-center",
        variant === "card" && "rounded-xl border border-border bg-card",
        className,
      )}
    >
      <Icon
        className={cn("mb-3 h-6 w-6 text-muted-foreground", iconClassName)}
        aria-hidden="true"
      />
      <p className="text-base font-semibold text-foreground">{title}</p>
      {body ? (
        <div className="mt-1 text-sm text-muted-foreground">{body}</div>
      ) : null}
      {action ? (
        <div className="mt-4 flex flex-wrap justify-center gap-2">{action}</div>
      ) : null}
    </div>
  );
}

export interface EmptyStateProps {
  icon?: IconType;
  title: ReactNode;
  body?: ReactNode;
  action?: ReactNode;
  variant?: "inline" | "card";
  className?: string;
}

/** Nothing to show yet (no results, no saved items). */
export function EmptyState({ icon = Inbox, ...props }: EmptyStateProps) {
  return <StateFrame icon={icon} role="status" {...props} />;
}

export interface ErrorStateProps {
  /** Error to describe; status codes map to human text. */
  error?: ErrorLike;
  /** Override the mapped title/body. */
  title?: ReactNode;
  body?: ReactNode;
  icon?: IconType;
  /** Shows a "Try again" button when the error is retryable. */
  onRetry?: () => void;
  /** Extra or replacement action. */
  action?: ReactNode;
  /** Where the sign-in action should return to (for 401/403). */
  signInHref?: string;
  variant?: "inline" | "card";
  className?: string;
}

/** A request failed. Replaces raw "Unavailable / Failed to fetch" cards. */
export function ErrorState({
  error,
  title,
  body,
  icon = AlertTriangle,
  onRetry,
  action,
  signInHref = "/login",
  variant,
  className,
}: ErrorStateProps) {
  const described = describeError(error);
  const actions = (
    <>
      {described.needsSignIn ? (
        <Button asChild size="sm">
          <Link to={signInHref}>Sign in</Link>
        </Button>
      ) : null}
      {onRetry && described.retryable ? (
        <Button type="button" variant="outline" size="sm" onClick={onRetry}>
          <RotateCw aria-hidden="true" />
          Try again
        </Button>
      ) : null}
      {action}
    </>
  );
  const hasActions =
    described.needsSignIn ||
    (onRetry && described.retryable) ||
    Boolean(action);

  return (
    <StateFrame
      icon={icon}
      iconClassName="text-destructive"
      role="alert"
      title={title ?? described.title}
      body={body ?? described.body}
      action={hasActions ? actions : undefined}
      variant={variant}
      className={className}
    />
  );
}
