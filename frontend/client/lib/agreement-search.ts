import { normalizeXmlText } from "@/components/XMLRenderer";

export interface AgreementTextSection {
  sectionUuid: string;
  articleTitle: string | null;
  sectionTitle: string | null;
  text: string;
}

const BODY_RE = /<body[^>]*>(.*?)<\/body>/s;
const ARTICLE_RE = /<article([^>]*)>(.*?)<\/article>/gs;
const SECTION_RE = /<section([^>]*)>(.*?)<\/section>/gs;
const TITLE_RE = /title="([^"]*)"/;
const UUID_RE = /uuid="([^"]*)"/;
const TAG_RE = /<[^>]+>/g;
const WS_RE = /\s+/g;

function attrValue(source: string, pattern: RegExp) {
  const match = source.match(pattern);
  return match?.[1] ?? null;
}

function stripXml(xml: string) {
  return normalizeXmlText(xml.replace(TAG_RE, " ").replace(WS_RE, " ").trim());
}

export function extractAgreementTextSections(xmlContent: string): AgreementTextSection[] {
  const bodyMatch = xmlContent.match(BODY_RE);
  const content = bodyMatch?.[1] ?? xmlContent;
  const sections: AgreementTextSection[] = [];

  for (const articleMatch of content.matchAll(ARTICLE_RE)) {
    const articleAttrs = articleMatch[1];
    const articleBody = articleMatch[2];
    const articleTitle = attrValue(articleAttrs, TITLE_RE);

    for (const sectionMatch of articleBody.matchAll(SECTION_RE)) {
      const sectionAttrs = sectionMatch[1];
      const sectionBody = sectionMatch[2];
      const sectionUuid = attrValue(sectionAttrs, UUID_RE);
      if (!sectionUuid) continue;
      sections.push({
        sectionUuid,
        articleTitle,
        sectionTitle: attrValue(sectionAttrs, TITLE_RE),
        text: stripXml(sectionBody),
      });
    }
  }

  return sections;
}
