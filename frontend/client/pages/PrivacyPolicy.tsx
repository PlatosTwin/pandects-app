import { LegalMarkdownPage } from "@/components/LegalMarkdownPage";
import markdown from "../../public/legal/privacy-policy.md?raw";

export default function PrivacyPolicy() {
  return (
    <LegalMarkdownPage
      title="Privacy Policy"
      downloadHref="/legal/privacy-policy.md"
      markdown={markdown}
    />
  );
}
