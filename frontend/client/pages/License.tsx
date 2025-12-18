import { Link } from "react-router-dom";
import { LegalMarkdownPage } from "@/components/LegalMarkdownPage";

export default function License() {
  return (
    <LegalMarkdownPage
      title="License"
      markdownPath="/legal/license.md"
      relatedLinks={
        <>
          <Link className="text-primary hover:underline" to="/terms">
            Terms of Service
          </Link>{" "}
          and{" "}
          <Link className="text-primary hover:underline" to="/privacy-policy">
            Privacy Policy
          </Link>
        </>
      }
    />
  );
}
