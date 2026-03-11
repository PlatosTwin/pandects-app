import { Link } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const NotFound = () => {
  return (
    <PageShell className="flex items-center justify-center py-16">
      <Card className="w-full max-w-lg p-8 text-center">
        <h1 className="text-4xl font-bold text-foreground">404</h1>
        <p className="mt-3 text-base text-muted-foreground">
          Page not found.
        </p>
        <div className="mt-6 flex flex-col justify-center gap-3 sm:flex-row">
          <Button asChild>
            <Link to="/">Return to Home</Link>
          </Button>
          <Button asChild variant="outline">
            <Link to="/search">Go to Search</Link>
          </Button>
        </div>
      </Card>
    </PageShell>
  );
};

export default NotFound;
