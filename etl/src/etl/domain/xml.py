import re
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
from dataclasses import dataclass


@dataclass
class XMLData:
    agreement_uuid: str
    xml: str


def get_uuid(x):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, x))


def convert_to_xml(
    tagged_text: str,
    agreement_uuid: str,
    acquirer: str,
    target: str,
    announcement_date: str,
    url: str,
    source_format: str,
) -> str:
    """
    Convert text with <article>...</article> and <section>...</section> headings
    into a proper XML hierarchy, wrapping any leading text (before the first <article>)
    in <recitals>/<text>/<definition>/<page> blocks, and wrapping all <article>…</article>
    elements inside a top-level <body> element. Standalone <page>…</page> lines become <page> elements.
    """
    # find all <article> or <section> headings and their positions
    pattern = re.compile(r"<(article|section)>(.*?)</\1>", re.DOTALL)
    matches = list(pattern.finditer(tagged_text))

    root = ET.Element("document", uuid=agreement_uuid)

    metadata = ET.SubElement(root, "metadata")
    ET.SubElement(metadata, "acquirer").text = acquirer
    ET.SubElement(metadata, "target").text = target
    ET.SubElement(metadata, "announcementDate").text = announcement_date
    ET.SubElement(metadata, "url").text = url
    ET.SubElement(metadata, "sourceFormat").text = source_format

    # Helper to add <text>, <definition>, <pageUUID> or <page> children
    def add_text_nodes(parent, text_block):
        # definition: starts with "…some text…" means …
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
                # extract the first quoted term, lowercase it
                m_term = term_re.match(stripped)
                term_val = m_term.group(1).lower() if m_term else ""

                # now include it as an attribute
                d = ET.SubElement(
                    parent, "definition", standardID="<placeholder>", term=term_val
                )
                d.text = stripped

            # 4) otherwise normal <text>
            else:
                t = ET.SubElement(parent, "text")
                t.text = stripped

    # 1) Leading text → <recitals>
    first_pos = matches[0].start() if matches else len(tagged_text)
    leading = tagged_text[:first_pos].strip()
    if leading:
        rec = ET.SubElement(root, "recitals")
        add_text_nodes(rec, leading)

    # 2) Create <body> wrapper and then process articles/sections into it
    body = ET.SubElement(root, "body")
    current_article = None
    section_count = 0

    for i, m in enumerate(matches):
        tag = m.group(1)
        raw_title = m.group(2).strip()
        title = " ".join(raw_title.split())

        start, end = m.span()
        next_start = (
            matches[i + 1].start() if i + 1 < len(matches) else len(tagged_text)
        )
        content = tagged_text[end:next_start].strip()

        if tag == "article":
            current_article = ET.SubElement(
                body,
                "article",
                title=title,
                uuid=get_uuid(agreement_uuid + title),
                standardId="<placeholder>",
            )
            section_count = 0
            if content:
                add_text_nodes(current_article, content)

        else:  # section
            if current_article is None:
                current_article = ET.SubElement(
                    body, "article", title="", uuid="", standardId=""
                )
                section_count = 0

            section_count += 1
            sec = ET.SubElement(
                current_article,
                "section",
                title=title,
                uuid=get_uuid(agreement_uuid + title + str(section_count)),
                order=str(section_count),
                standardId="<placeholder>",
            )
            if content:
                add_text_nodes(sec, content)

    # Pretty-print with encoding in header
    rough = ET.tostring(root, "utf-8")
    return rough


def collapse_text_into_definitions(xml_str: str) -> str:
    """
    1. Moves <text> (and the relevant <page>/<pageUUID>) into the preceding <definition>.
    2. Ensures that any free text directly inside a <definition> is wrapped in its own <text> tag,
       so <definition> contains only <text>, <page>, and <pageUUID> children.
    """
    root = ET.fromstring(xml_str)

    def process_container(parent):
        children = list(parent)
        for idx, child in enumerate(children):
            if child.tag != "definition":
                continue

            def_elem = child
            # locate next <definition> or end
            next_def_idx = next(
                (
                    k
                    for k in range(idx + 1, len(children))
                    if children[k].tag == "definition"
                ),
                len(children),
            )
            segment = children[idx + 1 : next_def_idx]

            # find any <text> in the segment
            text_indices = [i for i, el in enumerate(segment) if el.tag == "text"]
            if not text_indices:
                continue

            last_text_idx = max(text_indices)
            # choose elements to move: all <text>, plus any <page>/<pageUUID> at or before last text
            to_move = [
                el
                for i, el in enumerate(segment)
                if el.tag == "text"
                or (el.tag in ("page", "pageUUID") and i <= last_text_idx)
            ]

            # move them under the definition
            for el in to_move:
                parent.remove(el)
                def_elem.append(el)

    # First pass: collapse relevant <text>/<page>/<pageUUID> into definitions
    for elem in root.iter():
        process_container(elem)

    # Second pass: wrap any free-floating text inside <definition> into its own <text> tag
    for def_elem in root.iter("definition"):
        # wrap leading text
        if def_elem.text and def_elem.text.strip():
            txt = def_elem.text
            new = ET.Element("text")
            new.text = txt
            def_elem.insert(0, new)
        def_elem.text = None

        # wrap tails after children
        for child in list(def_elem):
            if child.tail and child.tail.strip():
                txt = child.tail
                new = ET.Element("text")
                new.text = txt
                idx = list(def_elem).index(child)
                def_elem.insert(idx + 1, new)
            child.tail = None

    return ET.tostring(root, encoding="unicode")


def generate_xml(df):
    staged_xml = []

    agreement_uuids = df["agreement_uuid"].unique().tolist()
    for agreement_uuid in agreement_uuids:
        temp = df[df["agreement_uuid"] == agreement_uuid].copy()

        url = temp["url"].to_list()[0]
        acquirer = temp["acquirer"].to_list()[0]
        target = temp["target"].to_list()[0]
        announcement_date = temp["date_announcement"].to_list()[0]
        if temp["is_html"].to_list()[0]:
            source_format = "html"
        else:
            source_format = "txt"

        temp["llm_output"] = temp[["llm_output", "page_uuid"]].apply(
            lambda x: x["llm_output"] + f"<pageUUID>{x['page_uuid']}</pageUUID>", axis=1
        )

        agreement_text = "\n".join(temp["llm_output"].values)

        xml_out = convert_to_xml(
            agreement_text,
            agreement_uuid,
            acquirer,
            target,
            announcement_date,
            url,
            source_format,
        )
        xml_out = collapse_text_into_definitions(xml_out)
        xml_out = xml.dom.minidom.parseString(xml_out).toprettyxml(indent="  ")

        staged_xml.append(XMLData(agreement_uuid=agreement_uuid, xml=xml_out))
        
    return staged_xml
