import { Component, type ErrorInfo, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { StatusPage } from "@/components/StatusPage";
import { Button } from "@/components/ui/button";
import { trackEvent } from "@/lib/analytics";
import { logger } from "@/lib/logger";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: (args: { error: Error; reset: () => void }) => ReactNode;
  scope?: string;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    const scope = this.props.scope ?? "unknown";
    logger.error(`ErrorBoundary[${scope}] caught:`, error, info.componentStack);
    trackEvent("react_error", {
      scope,
      message: error.message?.slice(0, 200) ?? "unknown",
      name: error.name ?? "Error",
    });
  }

  reset = () => {
    this.setState({ error: null });
  };

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    if (this.props.fallback) {
      return this.props.fallback({ error, reset: this.reset });
    }

    return <DefaultErrorFallback error={error} reset={this.reset} />;
  }
}

function DefaultErrorFallback({
  error,
  reset,
}: {
  error: Error;
  reset: () => void;
}) {
  const isDev = import.meta.env.DEV;
  return (
    <StatusPage
      role="alert"
      title="Something went wrong"
      description="An unexpected error interrupted this page. You can try again, or head back to the home page."
      actions={
        <>
          <Button onClick={reset}>Try again</Button>
          <Button asChild variant="outline">
            <Link to="/">Go home</Link>
          </Button>
        </>
      }
    >
      {isDev ? (
        <pre className="mt-6 max-h-48 overflow-auto rounded-md bg-muted/60 p-3 text-xs text-foreground/80">
          {error.name}: {error.message}
        </pre>
      ) : null}
    </StatusPage>
  );
}
