import { Link } from "react-router-dom";
import { LegalMarkdownPage } from "@/components/LegalMarkdownPage";
import markdown from "../../../LICENSE.md?raw";

function normalizeGplHtml(html: string): string {
  return html.replace(
    /(<h3 id="[^"]+">)(\d+\.\s+[^<.]+)(<\/h3>)/g,
    (_match, open: string, title: string, close: string) => `${open}${title}.${close}`,
  );
}

export default function SoftwareLicense() {
  return (
    <LegalMarkdownPage
      title="GPLv3 Software License"
      markdown={markdown}
      transformHtml={normalizeGplHtml}
      relatedLinks={
        <>
          <Link className="text-primary hover:underline" to="/license">
            License overview
          </Link>
          {", "}
          <Link className="text-primary hover:underline" to="/license/data">
            ODbL data license
          </Link>
          {", "}
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
