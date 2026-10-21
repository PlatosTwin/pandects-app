import { Link } from "react-router-dom";
import { StatusPage } from "@/components/StatusPage";
import { Button } from "@/components/ui/button";
import { useIsLockedOut } from "@/components/NavLinkParts";

const NotFound = () => {
  const isLockedOut = useIsLockedOut();
  return (
    <StatusPage
      code="404"
      title="Page not found."
      actions={
        <>
          <Button asChild>
            <Link to="/">Return to Home</Link>
          </Button>
          {!isLockedOut("/search") && (
            <Button asChild variant="outline">
              <Link to="/search">Go to Search</Link>
            </Button>
          )}
        </>
      }
    />
  );
};

export default NotFound;
