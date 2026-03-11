import { LegalMarkdownPage } from "@/components/LegalMarkdownPage";
import markdown from "../../public/legal/terms-of-service.md?raw";

export default function Terms() {
  return (
    <LegalMarkdownPage
      title="Terms of Service"
      markdownPath="/legal/terms-of-service.md"
      markdown={markdown}
    />
  );
}
