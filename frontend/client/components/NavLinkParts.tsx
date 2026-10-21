import { forwardRef, useContext, type ComponentPropsWithoutRef } from "react";
import { Link, matchPath } from "react-router-dom";
import { ExternalLink, Lock } from "lucide-react";
import { AuthContext } from "@/contexts/AuthContext";
import { cn } from "@/lib/utils";
import {
  LOCKDOWN_ACTIVE,
  isGatedLink,
  type NavLinkItem,
} from "@/components/navigation-links";
import { ACCESS_GATED_PATHS } from "@/lib/lockdown";

// Reads auth state the way <AccessGate> does, but tolerates a missing
// provider (static renders, tests) by treating it as still loading.
export function useNavAuth() {
  const ctx = useContext(AuthContext);
  return {
    status: ctx?.status ?? "loading",
    user: ctx?.user ?? null,
    logout: ctx?.logout,
  };
}

// True when a link would land a signed-out visitor on the lockdown notice.
// Hidden while auth is resolving so the glyph does not flash for members.
export function useShowLock(): (link: NavLinkItem) => boolean {
  const { status } = useNavAuth();
  const locked = LOCKDOWN_ACTIVE && status === "anonymous";
  return (link) => locked && isGatedLink(link);
}

// True when `to` would show this visitor the lockdown notice. For browse
// links and calls to action, which are hidden rather than marked with a lock.
// "Still resolving" counts as locked out: prerendered pages render that way,
// so a signed-out visitor never sees a link appear and then vanish, and a
// member sees it appear once their account is confirmed.
export function useIsLockedOut(): (to: string) => boolean {
  const { status } = useNavAuth();
  const locked = LOCKDOWN_ACTIVE && status !== "authenticated";
  return (to) => locked && isGatedPath(to);
}

/** Whether a URL ("/agreements/abc?from=/") opens a gated route pattern. */
export function isGatedPath(to: string): boolean {
  const pathname = to.split(/[?#]/)[0] || "/";
  return [...ACCESS_GATED_PATHS].some(
    (pattern) => matchPath({ path: pattern, end: true }, pathname) !== null,
  );
}

export function NavLinkContent({
  link,
  locked = false,
}: {
  link: NavLinkItem;
  locked?: boolean;
}) {
  return (
    <>
      <span>{link.label}</span>
      {locked && (
        <>
          <Lock
            className="h-3 w-3 shrink-0 text-muted-foreground"
            aria-hidden="true"
            data-testid="nav-lock"
          />
          <span className="sr-only">(sign-in required)</span>
        </>
      )}
      {link.kind === "external" && (
        <>
          <ExternalLink
            className="h-3 w-3 shrink-0 text-muted-foreground"
            aria-hidden="true"
          />
          <span className="sr-only">(opens in a new tab)</span>
        </>
      )}
    </>
  );
}

type NavItemLinkProps = Omit<ComponentPropsWithoutRef<"a">, "href"> & {
  link: NavLinkItem;
  active?: boolean;
  locked?: boolean;
};

// Renders an internal route as <Link> and an external URL as a new-tab <a>.
// Forwards refs and props so it works under Radix `asChild` slots.
export const NavItemLink = forwardRef<HTMLAnchorElement, NavItemLinkProps>(
  function NavItemLink(
    { link, active = false, locked = false, children, ...rest },
    ref,
  ) {
    const content = children ?? <NavLinkContent link={link} locked={locked} />;
    if (link.kind === "external") {
      return (
        <a
          ref={ref}
          href={link.href}
          target="_blank"
          rel="noopener noreferrer"
          data-panda-target={link.pandaTarget}
          {...rest}
        >
          {content}
        </a>
      );
    }
    return (
      <Link
        ref={ref}
        to={link.to}
        data-panda-target={link.pandaTarget}
        aria-current={active ? "page" : undefined}
        {...rest}
      >
        {content}
      </Link>
    );
  },
);

export function avatarInitial(email: string): string {
  const first = email.trim().charAt(0);
  return first ? first.toUpperCase() : "?";
}

export function AvatarInitial({
  email,
  className,
}: {
  email: string;
  className?: string;
}) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary ring-1 ring-inset ring-primary/20",
        className,
      )}
    >
      {avatarInitial(email)}
    </span>
  );
}
