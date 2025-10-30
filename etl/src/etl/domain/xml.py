import re
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import date


@dataclass
class XMLData:
    """Data structure for XML output."""

    agreement_uuid: str
    xml: str


def get_uuid(x: str) -> str:
    """Generate a UUID5 hash from the input string."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, x))


def convert_to_xml(
    tagged_text: str,
    agreement_uuid: str,
    acquirer: str,
    target: str,
    filing_date: date,
    url: str,
    source_format: str,
) -> str:
    """
    Convert text with <article>...</article> and <section>...</section> headings
    into a proper XML hierarchy.

    Process:
    - Wraps any leading text (before the first <article>) in <recitals>/<text>/<definition>/<page> blocks
    - Wraps all <article>…</article> elements inside a top-level <body> element
    - Standalone <page>…</page> lines become <page> elements

    Args:
        tagged_text: Text containing article and section tags.
        agreement_uuid: UUID of the agreement.
        acquirer: Name of the acquiring company.
        target: Name of the target company.
        announcement_date: Date of the announcement.
        url: URL of the source document.
        source_format: Format of the source document.

    Returns:
        XML string representation of the document.
    """
    # Find all <article> or <section> headings and their positions
    pattern = re.compile(r"<(article|section)>(.*?)</\1>", re.DOTALL)
    matches = list(pattern.finditer(tagged_text))

    root = ET.Element("document", uuid=agreement_uuid)

    # Add metadata
    metadata = ET.SubElement(root, "metadata")
    ET.SubElement(metadata, "acquirer").text = acquirer
    ET.SubElement(metadata, "target").text = target
    ET.SubElement(metadata, "filingDate").text = filing_date.strftime("%Y-%m-%d")
    ET.SubElement(metadata, "url").text = url
    ET.SubElement(metadata, "sourceFormat").text = source_format

    # Helper to add <text>, <definition>, <pageUUID> or <page> children
    def add_text_nodes(parent: ET.Element, text_block: str) -> None:
        """
        Add text nodes to the parent element based on content patterns.

        Args:
            parent: Parent XML element.
            text_block: Text content to process.
        """
        # Definition: starts with "…some text…" means …
        definition_re_a = re.compile(
            r'^[\u201C\u201D"]'  # opening curly or straight quote
            r'[^"\u201C\u201D]+'  # the term itself
            r'[\u201C\u201D"]\s+'  # closing quote + space
            r"(?:mean|means|shall have the meaning|shall mean)\b",
            re.IGNORECASE,
        )
        term_re = re.compile(r'^[\u201C\u201D"]([^"\u201C\u201D]+)[\u201C\u201D"]')

        definition_re_b = re.compile(
            r"""(?xi)                       # case‐insensitive, verbose
            (?:                             # two alternatives:
              [\u201C\u201D"]               #   opening curly or straight quote
              ([^"\u201C\u201D]+)           #   term1
              [\u201C\u201D"]               #   closing quote
              \s+or\s+                      
              [\u201C\u201D"]               #   opening quote for term2
              ([^"\u201C\u201D]+)           #   term2
              [\u201C\u201D"]               #   closing quote
              \s+
              (?:mean|means|shall\ mean|shall\ have\ the\ meaning)
            |
              [\u201C\u201D"]               #   opening quote
              ([^"\u201C\u201D]+)           #   term
              [\u201C\u201D"]               #   closing quote
              (?:\s+\S+){0,5}               #   up to 4 words
              \s+
              (?:mean|means|shall\ mean|shall\ have\ the\ meaning)
            )
            \b""",
            re.IGNORECASE | re.VERBOSE,
        )

        for line in text_block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # 1) <pageUUID>
            if "<pageUUID>" in stripped and "</pageUUID>" in stripped:
                m_uuid = re.search(r"<pageUUID>(.*?)</pageUUID>", stripped)
                pu = ET.SubElement(parent, "pageUUID")
                pu.text = m_uuid.group(1).strip() if m_uuid else stripped

            # 2) <page>…</page>
            elif "<page>" in stripped and "</page>" in stripped:
                m_page = re.search(r"<page>(.*?)</page>", stripped)
                p = ET.SubElement(parent, "page")
                p.text = m_page.group(1).strip() if m_page else stripped

            # 3) <definition> if it starts with "…" means …
            elif definition_re_a.match(stripped) or definition_re_b.match(stripped):
                # Extract the first quoted term, lowercase it
                m_term = term_re.match(stripped)
                term_val = m_term.group(1).lower() if m_term else ""

                # Now include it as an attribute
                d = ET.SubElement(
                    parent, "definition", standardID="<placeholder>", term=term_val
                )
                d.text = stripped

            # 4) Otherwise normal <text>
            else:
                t = ET.SubElement(parent, "text")
                t.text = stripped

    # 1) Leading text → <frontMatter>
    first_pos = matches[0].start() if matches else len(tagged_text)
    leading = tagged_text[:first_pos].strip()
    if leading:
        rec = ET.SubElement(root, "frontMatter")
        add_text_nodes(rec, leading)

    # 2) Create <body> wrapper and then process articles/sections into it
    body = ET.SubElement(root, "body")
    current_article = None
    section_count = 0
    article_count = 0

    for i, m in enumerate(matches):
        tag = m.group(1)
        raw_title = m.group(2).strip()
        title = " ".join(raw_title.split())

        start, end = m.span()
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(tagged_text)
        content = tagged_text[end:next_start].strip()

        if tag == "article":
            article_count += 1

            current_article = ET.SubElement(
                body,
                "article",
                title=title,
                uuid=get_uuid(agreement_uuid + title),
                order=str(article_count),
                standardId="<placeholder>",
            )

            section_count = 0
            if content:
                add_text_nodes(current_article, content)

        else:  # section
            # If no current article, attach section directly under body.
            container = current_article if current_article is not None else body

            section_count += 1
            sec = ET.SubElement(
                container,
                "section",
                title=title,
                uuid=get_uuid(agreement_uuid + title + str(section_count)),
                order=str(section_count),
                standardId="<placeholder>",
            )
            if content:
                add_text_nodes(sec, content)

    # Note: trailing text after the final heading remains within the last article/section's content.

    # Pretty-print with encoding in header
    rough = ET.tostring(root, "utf-8")
    return rough


def collapse_text_into_definitions(xml_str: str) -> str:
    """
    Move <text> elements into preceding <definition> elements and ensure proper structure.

    Process:
    1. Moves <text> (and the relevant <page>/<pageUUID>) into the preceding <definition>.
    2. Ensures that any free text directly inside a <definition> is wrapped in its own <text> tag,
       so <definition> contains only <text>, <page>, and <pageUUID> children.

    Args:
        xml_str: XML string to process.

    Returns:
        Processed XML string.
    """
    root = ET.fromstring(xml_str)

    def process_container(parent: ET.Element) -> None:
        """Process a container element to move text into definitions."""
        children = list(parent)
        for idx, child in enumerate(children):
            if child.tag != "definition":
                continue

            def_elem = child
            # Locate next <definition> or end
            next_def_idx = next(
                (
                    k
                    for k in range(idx + 1, len(children))
                    if children[k].tag == "definition"
                ),
                len(children),
            )
            segment = children[idx + 1 : next_def_idx]

            # Find any <text> in the segment
            text_indices = [i for i, el in enumerate(segment) if el.tag == "text"]
            if not text_indices:
                continue

            last_text_idx = max(text_indices)
            # Choose elements to move: all <text>, plus any <page>/<pageUUID> at or before last text
            to_move = [
                el
                for i, el in enumerate(segment)
                if el.tag == "text"
                or (el.tag in ("page", "pageUUID") and i <= last_text_idx)
            ]

            # Move them under the definition
            for el in to_move:
                parent.remove(el)
                def_elem.append(el)

    # First pass: collapse relevant <text>/<page>/<pageUUID> into definitions
    for elem in root.iter():
        process_container(elem)

    # Second pass: wrap any free-floating text inside <definition> into its own <text> tag
    for def_elem in root.iter("definition"):
        # Wrap leading text
        if def_elem.text and def_elem.text.strip():
            txt = def_elem.text
            new = ET.Element("text")
            new.text = txt
            def_elem.insert(0, new)
        def_elem.text = None

        # Wrap tails after children
        for child in list(def_elem):
            if child.tail and child.tail.strip():
                txt = child.tail
                new = ET.Element("text")
                new.text = txt
                idx = list(def_elem).index(child)
                def_elem.insert(idx + 1, new)
            child.tail = None

    return ET.tostring(root, encoding="unicode")


def generate_xml(df: Any) -> List[XMLData]:
    """
    Generate XML data from a DataFrame.

    Args:
        df: DataFrame containing agreement data.

    Returns:
        List of XMLData objects.
    """
    staged_xml = []

    # Helper: add simple nodes (text/definition/page/pageUUID) to container
    def add_text_nodes_simple(parent: ET.Element, text_block: str) -> None:
        definition_re_a = re.compile(
            r'^[\u201C\u201D\"][^"\u201C\u201D]+[\u201C\u201D\"]\s+(?:mean|means|shall have the meaning|shall mean)\b',
            re.IGNORECASE,
        )
        term_re = re.compile(r'^[\u201C\u201D\"]([^"\u201C\u201D]+)[\u201C\u201D\"]')
        definition_re_b = re.compile(
            r"""(?xi)
            (?:
              [\u201C\u201D\"][^"\u201C\u201D]+[\u201C\u201D\"]\s+or\s+[\u201C\u201D\"][^"\u201C\u201D]+[\u201C\u201D\"]\s+(?:mean|means|shall\ mean|shall\ have\ the\ meaning)
            |
              [\u201C\u201D\"][^"\u201C\u201D]+[\u201C\u201D\"](?:\s+\S+){0,5}\s+(?:mean|means|shall\ mean|shall\ have\ the\ meaning)
            )\b
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        for line in text_block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "<pageUUID>" in stripped and "</pageUUID>" in stripped:
                m_uuid = re.search(r"<pageUUID>(.*?)</pageUUID>", stripped)
                pu = ET.SubElement(parent, "pageUUID")
                pu.text = m_uuid.group(1).strip() if m_uuid else stripped
            elif "<page>" in stripped and "</page>" in stripped:
                m_page = re.search(r"<page>(.*?)</page>", stripped)
                p = ET.SubElement(parent, "page")
                p.text = m_page.group(1).strip() if m_page else stripped
            elif definition_re_a.match(stripped) or definition_re_b.match(stripped):
                m_term = term_re.match(stripped)
                term_val = m_term.group(1).lower() if m_term else ""
                d = ET.SubElement(parent, "definition", standardID="<placeholder>", term=term_val)
                d.text = stripped
            else:
                t = ET.SubElement(parent, "text")
                t.text = stripped

    agreement_uuids = df["agreement_uuid"].unique().tolist()
    for agreement_uuid in agreement_uuids:
        temp = df[df["agreement_uuid"] == agreement_uuid].copy()
        # Preserve order
        if "page_order" in temp.columns:
            temp = temp.sort_values(by=["page_order", "page_uuid"], kind="stable")
        else:
            temp = temp.sort_values(by=["page_uuid"], kind="stable")

        url = temp["url"].to_list()[0]
        acquirer = temp["acquirer"].to_list()[0]
        target = temp["target"].to_list()[0]
        announcement_date = temp["filing_date"].to_list()[0]
        source_format = "html" if temp["source_is_html"].to_list()[0] else "txt"

        root = ET.Element("document", uuid=agreement_uuid)
        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "acquirer").text = acquirer
        ET.SubElement(metadata, "target").text = target
        ET.SubElement(metadata, "filingDate").text = announcement_date.strftime("%Y-%m-%d")
        ET.SubElement(metadata, "url").text = url
        ET.SubElement(metadata, "sourceFormat").text = source_format

        # Containers by page type
        # frontMatter
        fm_rows = temp[temp.get("source_page_type") == "front_matter"]
        if not fm_rows.empty:
            fm_el = ET.SubElement(root, "frontMatter")
            text_block = "\n".join(
                (r["tagged_output"] + f"<pageUUID>{r['page_uuid']}</pageUUID>") for _, r in fm_rows.iterrows()
            )
            add_text_nodes_simple(fm_el, text_block)

        # tableOfContents
        toc_rows = temp[temp.get("source_page_type") == "toc"]
        if not toc_rows.empty:
            toc_el = ET.SubElement(root, "tableOfContents")
            text_block = "\n".join(
                (r["tagged_output"] + f"<pageUUID>{r['page_uuid']}</pageUUID>") for _, r in toc_rows.iterrows()
            )
            add_text_nodes_simple(toc_el, text_block)

        # body (preserve page order; parse headings across all body pages)
        body_rows = temp[temp.get("source_page_type") == "body"]
        if not body_rows.empty:
            body_el = ET.SubElement(root, "body")
            body_text = "\n".join(
                (r["tagged_output"] + f"<pageUUID>{r['page_uuid']}</pageUUID>") for _, r in body_rows.iterrows()
            )
            tmp_xml = convert_to_xml(
                body_text,
                agreement_uuid,
                acquirer,
                target,
                announcement_date,
                url,
                source_format,
            )
            tmp_root = ET.fromstring(tmp_xml)
            # Include any leading content that appeared before the first heading within body pages
            fm_tmp = tmp_root.find("frontMatter")
            if fm_tmp is not None:
                for child in list(fm_tmp):
                    body_el.append(child)
            # Merge parsed body (articles/sections spanning pages)
            body_tmp = tmp_root.find("body")
            if body_tmp is not None:
                for child in list(body_tmp):
                    body_el.append(child)

        # sigPages
        sig_rows = temp[temp.get("source_page_type") == "sig"]
        if not sig_rows.empty:
            sig_el = ET.SubElement(root, "sigPages")
            text_block = "\n".join(
                (r["tagged_output"] + f"<pageUUID>{r['page_uuid']}</pageUUID>") for _, r in sig_rows.iterrows()
            )
            add_text_nodes_simple(sig_el, text_block)

        # backMatter
        bm_rows = temp[temp.get("source_page_type") == "back_matter"]
        if not bm_rows.empty:
            bm_el = ET.SubElement(root, "backMatter")
            text_block = "\n".join(
                (r["tagged_output"] + f"<pageUUID>{r['page_uuid']}</pageUUID>") for _, r in bm_rows.iterrows()
            )
            add_text_nodes_simple(bm_el, text_block)

        xml_str = ET.tostring(root, encoding="unicode")
        xml_str = collapse_text_into_definitions(xml_str)
        xml_str = xml.dom.minidom.parseString(xml_str).toprettyxml(indent="  ")
        staged_xml.append(XMLData(agreement_uuid=agreement_uuid, xml=xml_str))

    return staged_xml
