# alto_parser.py
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Iterator, Tuple

# Namespace used in your ALTO files
ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}


def extract_text_from_alto_file(xml_path: Path) -> str:
    """
    Extract full text from a single ALTO v4 PAGE-XML file.
    
    Reading order:
        TextBlock -> TextLine -> (String/SP/HYP) in document order.
    
    Hyphenation handling:
        - If a String has SUBS_TYPE="HypPart1" and SUBS_CONTENT, we use SUBS_CONTENT
          and skip the following HypPart2.
        - HYP elements (soft hyphen markers) are ignored.
    
    Spaces:
        - SP elements become literal spaces " ".
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    page_lines = []

    # Iterate over text blocks in document order
    for textblock in root.findall(".//alto:TextBlock", ALTO_NS):
        # Within each block, iterate over lines
        for line in textblock.findall("alto:TextLine", ALTO_NS):
            tokens = []
            skip_next_hyp_part2 = False

            # Children of TextLine are typically String, SP, HYP
            for child in list(line):
                tag = child.tag
                if tag.endswith("String"):
                    subs_type = child.get("SUBS_TYPE")
                    subs_content = child.get("SUBS_CONTENT")
                    content = child.get("CONTENT", "")

                    if subs_type == "HypPart2" and skip_next_hyp_part2:
                        # We already used SUBS_CONTENT from HypPart1
                        skip_next_hyp_part2 = False
                        continue

                    if subs_type == "HypPart1" and subs_content:
                        # Use the full word and ignore HypPart2
                        tokens.append(subs_content)
                        skip_next_hyp_part2 = True
                    else:
                        tokens.append(content)

                elif tag.endswith("SP"):
                    # Explicit space
                    tokens.append(" ")

                elif tag.endswith("HYP"):
                    # Soft hyphen marker: ignore in reconstructed text
                    continue

            # Merge tokens for the line
            line_text = "".join(tokens).strip()
            if line_text:
                page_lines.append(line_text)

    # Join lines with newlines
    return "\n".join(page_lines)


def iter_alto_pages(root_dir: Path) -> Iterator[Tuple[str, Path, str]]:
    """
    Walk a directory tree of 'volume' folders containing ALTO XML files
    and yield (volume_id, xml_path, page_text) for each page.
    
    Assumes:
        root_dir/
            VOL0001/
                30002022_00001.xml
                30002022_00002.xml
                ...
            VOL0002/
                ...
    Where 'volume_id' is taken to be the name of the immediate parent directory.
    """
    for xml_path in root_dir.rglob("*.xml"):
        if not xml_path.is_file():
            continue
        volume_id = xml_path.parent.name
        text = extract_text_from_alto_file(xml_path)
        yield volume_id, xml_path, text

