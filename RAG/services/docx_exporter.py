"""Convert edited workspace Markdown to a formatted Word document (.docx).

Translates headings, lists, basic inline styling (bold, italic, underline, strike,
highlight), inline code, external links, tables, and embeds local figures correctly.
"""
from pathlib import Path
import markdown as md_lib
from bs4 import BeautifulSoup
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_COLOR_INDEX

from RAG.services import paths


def render(source: str) -> Path:
    """Render workspace markdown to a DOCX file and return its path."""
    md_path = paths.workspace_md_path(source)
    if not md_path.exists():
        raise FileNotFoundError(f"Workspace markdown not found: {md_path}")

    paths.ensure_dirs()
    text = md_path.read_text(encoding="utf-8")

    # Render markdown to HTML body
    html_body = md_lib.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "nl2br"],
    )

    doc = Document()
    
    # Configure default styles briefly
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)

    soup = BeautifulSoup(html_body, "html.parser")

    # Walk children of the HTML body
    for el in soup.children:
        if el.name is None:
            continue

        # Headings
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(el.name[1])
            doc.add_heading(el.text, level=level)

        # Paragraphs
        elif el.name == "p":
            p = doc.add_paragraph()
            _process_inline(p, el, source)

        # Lists (ul, ol)
        elif el.name in ("ul", "ol"):
            list_style = "List Bullet" if el.name == "ul" else "List Number"
            for li in el.find_all("li", recursive=False):
                p = doc.add_paragraph(style=list_style)
                _process_inline(p, li, source)

        # Blockquotes
        elif el.name == "blockquote":
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.5)
            _process_inline(p, el, source)
            # Make the blockquote italic
            for run in p.runs:
                run.italic = True

        # Tables
        elif el.name == "table":
            rows = el.find_all("tr")
            if not rows:
                continue
            
            # Compute max columns
            max_cols = 1
            for row in rows:
                cols = len(row.find_all(["td", "th"]))
                if cols > max_cols:
                    max_cols = cols

            table = doc.add_table(rows=len(rows), cols=max_cols)
            table.style = "Table Grid"

            for r_idx, row in enumerate(rows):
                cells = row.find_all(["td", "th"])
                for c_idx, cell in enumerate(cells):
                    if c_idx < max_cols:
                        table.cell(r_idx, c_idx).text = cell.text

        # Fenced code/pre
        elif el.name == "pre":
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.5)
            code = el.find("code")
            txt = code.text if code else el.text
            run = p.add_run(txt)
            run.font.name = "Courier New"
            run.font.size = Pt(10)

        # Horizontal rules
        elif el.name == "hr":
            doc.add_paragraph("---")

    out_path = paths.workspace_docx_path(source)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return out_path


def _process_inline(p, el, source: str):
    """Recursively process HTML children within an element and append runs to paragraph p."""
    for child in el.contents:
        if isinstance(child, str):
            # Text node
            if child.strip() == "" and not child.startswith(" ") and not child.endswith(" "):
                continue
            p.add_run(child)
        elif child.name in ("strong", "b"):
            run = p.add_run(child.text)
            run.bold = True
        elif child.name in ("em", "i"):
            run = p.add_run(child.text)
            run.italic = True
        elif child.name == "u":
            run = p.add_run(child.text)
            run.underline = True
        elif child.name == "mark":
            run = p.add_run(child.text)
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif child.name in ("strike", "del", "s"):
            run = p.add_run(child.text)
            run.font.strike = True
        elif child.name == "code":
            run = p.add_run(child.text)
            run.font.name = "Courier New"
            run.font.size = Pt(10)
        elif child.name == "a":
            # Render a blue/underlined text followed by the url
            run = p.add_run(child.text)
            run.font.underline = True
            run.font.color.rgb = None  # standard blue/theme color is default
            href = child.get("href", "")
            if href:
                p.add_run(f" ({href})")
        elif child.name == "img":
            src = child.get("src", "")
            if src.startswith("/images/workspace/") or src.startswith("/images/originals/"):
                parts = src.strip("/").split("/")
                if len(parts) >= 4:
                    channel = parts[1]
                    stem = parts[2]
                    img_name = parts[3]
                    img_path = paths.image_path(channel, stem, img_name)
                    if img_path and img_path.exists():
                        # Add image on a new run
                        try:
                            p.add_run().add_picture(str(img_path), width=Inches(4.5))
                        except Exception as e:
                            print(f"[DOCXExporter] Failed to embed image {img_path}: {e}")
        else:
            # Fallback recurse
            _process_inline(p, child, source)
