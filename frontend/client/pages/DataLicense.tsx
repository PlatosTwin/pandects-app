import { Link } from "react-router-dom";

import { LegalMarkdownPage } from "@/components/LegalMarkdownPage";

import markdown from "../legal/odbl-1.0.md?raw";

export default function DataLicense() {
  return (
    <LegalMarkdownPage
      title="ODbL Data License"
      markdown={markdown}
      relatedLinks={
        <>
          <Link className="text-primary hover:underline" to="/license">
            License overview
          </Link>
          {", "}
          <Link className="text-primary hover:underline" to="/license/software">
            GPLv3 software license
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
