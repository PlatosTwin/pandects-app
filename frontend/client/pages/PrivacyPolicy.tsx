import { LegalMarkdownPage } from "@/components/LegalMarkdownPage";
import markdown from "../../public/legal/privacy-policy.md?raw";

export default function PrivacyPolicy() {
  return (
    <LegalMarkdownPage
      title="Privacy Policy"
      markdownPath="/legal/privacy-policy.md"
      markdown={markdown}
    />
  );
}
