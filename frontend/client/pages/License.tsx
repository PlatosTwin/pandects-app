import { LegalMarkdownPage } from "@/components/LegalMarkdownPage";
import markdown from "../../public/legal/license.md?raw";

export default function License() {
  return (
    <LegalMarkdownPage
      title="License"
      downloadHref="/legal/license.md"
      markdown={markdown}
    />
  );
}
