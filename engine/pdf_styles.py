"""Shared styles and document components for the FinSight AI PDF report.

Professional research-report look: white background, navy/charcoal
headings, subtle gray tables, one restrained accent color.
"""

from copy import deepcopy

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ------------------------------------------------------------------
# Palette & page geometry
# ------------------------------------------------------------------

NAVY = colors.HexColor("#1F2A44")
CHARCOAL = colors.HexColor("#333A45")
ACCENT = colors.HexColor("#2F6FED")
ACCENT_LIGHT = colors.HexColor("#EAF1FE")
GRAY_LINE = colors.HexColor("#D4D9E0")
GRAY_BG = colors.HexColor("#F4F6F9")
GREEN = colors.HexColor("#1E7F4F")
RED = colors.HexColor("#B3372F")
AMBER = colors.HexColor("#9A6B0F")

PAGE_SIZE = A4
MARGIN = 20 * mm

# Try a bundled TTF so glyph coverage is predictable; fall back to
# Helvetica (WinAnsi — note: no rupee glyph, so the report uses "Rs").
try:
    import matplotlib

    _mpl_data = matplotlib.matplotlib_fname()
    import os as _os

    _dejavu = _os.path.join(
        _os.path.dirname(_mpl_data), "fonts", "ttf", "DejaVuSans.ttf"
    )
    _dejavu_bold = _os.path.join(
        _os.path.dirname(_mpl_data), "fonts", "ttf", "DejaVuSans-Bold.ttf"
    )
    pdfmetrics.registerFont(TTFont("FinSans", _dejavu))
    pdfmetrics.registerFont(TTFont("FinSans-Bold", _dejavu_bold))
    BODY_FONT = "FinSans"
    BOLD_FONT = "FinSans-Bold"
except Exception:
    BODY_FONT = "Helvetica"
    BOLD_FONT = "Helvetica-Bold"


def get_styles():
    """Report paragraph styles (TOC hooks into Heading1/Heading2)."""
    base = getSampleStyleSheet()

    styles = {
        "cover_title": ParagraphStyle(
            "cover_title", parent=base["Title"], fontName=BOLD_FONT,
            fontSize=30, leading=34, textColor=NAVY, alignment=1,
            spaceAfter=2 * mm,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", parent=base["Normal"], fontName=BODY_FONT,
            fontSize=12, leading=15, textColor=CHARCOAL, alignment=1,
            spaceAfter=8 * mm,
        ),
        "cover_field": ParagraphStyle(
            "cover_field", parent=base["Normal"], fontName=BODY_FONT,
            fontSize=10, leading=14, textColor=CHARCOAL, alignment=1,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName=BOLD_FONT,
            fontSize=16, leading=19, textColor=NAVY,
            spaceBefore=6 * mm, spaceAfter=3 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName=BOLD_FONT,
            fontSize=12, leading=15, textColor=NAVY,
            spaceBefore=4 * mm, spaceAfter=2 * mm,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"], fontName=BOLD_FONT,
            fontSize=10.5, leading=13, textColor=CHARCOAL,
            spaceBefore=3 * mm, spaceAfter=1.5 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName=BODY_FONT,
            fontSize=9.5, leading=13.5, textColor=CHARCOAL,
            spaceAfter=2 * mm, alignment=4,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"], fontName=BODY_FONT,
            fontSize=9.5, leading=13.5, textColor=CHARCOAL,
            leftIndent=8 * mm, bulletIndent=3 * mm,
            spaceAfter=1.2 * mm,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontName=BODY_FONT,
            fontSize=8, leading=10.5, textColor=colors.HexColor("#6B7280"),
            spaceBefore=1 * mm, spaceAfter=3 * mm, alignment=1,
        ),
        "cell": ParagraphStyle(
            "cell", parent=base["Normal"], fontName=BODY_FONT,
            fontSize=8.5, leading=11, textColor=CHARCOAL,
        ),
        "cell_bold": ParagraphStyle(
            "cell_bold", parent=base["Normal"], fontName=BOLD_FONT,
            fontSize=8.5, leading=11, textColor=CHARCOAL,
        ),
        "cell_right": ParagraphStyle(
            "cell_right", parent=base["Normal"], fontName=BODY_FONT,
            fontSize=8.5, leading=11, textColor=CHARCOAL, alignment=2,
        ),
        "cell_header": ParagraphStyle(
            "cell_header", parent=base["Normal"], fontName=BOLD_FONT,
            fontSize=8.5, leading=11, textColor=colors.white,
        ),
        "toc": ParagraphStyle(
            "toc", parent=base["Normal"], fontName=BODY_FONT,
            fontSize=10, leading=15, textColor=CHARCOAL,
        ),
        "footer": ParagraphStyle(
            "footer", parent=base["Normal"], fontName=BODY_FONT,
            fontSize=7.5, leading=9, textColor=colors.HexColor("#6B7280"),
            alignment=1,
        ),
        "chat_user": ParagraphStyle(
            "chat_user", parent=base["Normal"], fontName=BODY_FONT,
            fontSize=9.5, leading=13.5, textColor=CHARCOAL,
        ),
        "chat_ai": ParagraphStyle(
            "chat_ai", parent=base["Normal"], fontName=BODY_FONT,
            fontSize=9.5, leading=13.5, textColor=colors.HexColor("#1E293B"),
        ),
        "chat_label_user": ParagraphStyle(
            "chat_label_user", parent=base["Normal"], fontName=BOLD_FONT,
            fontSize=9, leading=12, textColor=colors.HexColor("#6B7280"),
        ),
        "chat_label_ai": ParagraphStyle(
            "chat_label_ai", parent=base["Normal"], fontName=BOLD_FONT,
            fontSize=9, leading=12, textColor=ACCENT,
        ),
    }
    return styles


# ------------------------------------------------------------------
# Reusable components
# ------------------------------------------------------------------

def spacer(height_mm=4):
    return Spacer(1, height_mm * mm)


def rule():
    return HRFlowable(
        width="100%", thickness=0.6, color=GRAY_LINE,
        spaceBefore=2 * mm, spaceAfter=3 * mm,
    )


def section_heading(text, styles, level=1, break_before=True):
    """Heading flowables; h1/h2 feed the table of contents."""
    items = []
    if break_before:
        items.append(PageBreak())
    style = styles["h1"] if level == 1 else styles["h2"]
    items.append(Paragraph(text, style))
    items.append(rule())
    return items


def styled_table(headers, rows, styles, col_widths=None, repeat_header=True):
    """Gray-bordered table with navy header; spans pages with repeat."""
    header_cells = [Paragraph(str(h), styles["cell_header"]) for h in headers]
    body = []
    for row in rows:
        body.append([
            cell if hasattr(cell, "wrap")
            else Paragraph(str(cell), styles["cell"])
            for cell in row
        ])
    data = [header_cells] + body
    table = Table(data, colWidths=col_widths, repeatRows=1 if repeat_header else 0)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), BOLD_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, GRAY_LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRAY_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def kpi_grid(items, styles, cols=4):
    """Compact KPI cards as a bordered table: [(label, value), ...]."""
    rows = []
    for i in range(0, len(items), cols):
        chunk = items[i:i + cols]
        label_row = [Paragraph(f"<b>{label}</b>", styles["cell"]) for label, _ in chunk]
        value_row = [
            Paragraph(f"<font size=11><b>{value}</b></font>", styles["cell"])
            for _, value in chunk
        ]
        rows.append(label_row)
        rows.append(value_row)
    table = Table(rows)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, GRAY_LINE),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def callout(text, styles, tone="info"):
    """Tinted callout box for findings, notes and AI text."""
    bg = {
        "info": ACCENT_LIGHT,
        "success": colors.HexColor("#EAF7F0"),
        "warn": colors.HexColor("#FDF3E3"),
        "risk": colors.HexColor("#FDECEC"),
    }.get(tone, ACCENT_LIGHT)
    inner = [[Paragraph(text, styles["body"])]]
    box = Table(inner, colWidths=[170 * mm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.5, GRAY_LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    return box


def chat_message(role, text, meta, styles):
    """USER vs FINSIGHT AI message block with distinct formatting."""
    is_user = role == "USER"
    label_style = styles["chat_label_user"] if is_user else styles["chat_label_ai"]
    body_style = styles["chat_user"] if is_user else styles["chat_ai"]
    bg = GRAY_BG if is_user else colors.white
    border = GRAY_LINE if is_user else ACCENT
    header = f"{role}" + (f" &nbsp;&nbsp;·&nbsp;&nbsp; {meta}" if meta else "")
    inner = [
        [Paragraph(header, label_style)],
        [Paragraph(text, body_style)],
    ]
    box = Table(inner, colWidths=[170 * mm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.8, border),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, border),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    return box


class ReportCanvas:
    """Header/footer + Page X of Y via two-pass rendering.

    Usage: canvasmaker = ReportCanvas.make(header_left, header_right,
    footer_left, footer_mid). Build once into a throwaway buffer to count
    pages, set canvasmaker.set_total(n), then build the real output.
    """

    @classmethod
    def make(cls, header_left, header_right, footer_left, footer_mid):
        """Return a canvasmaker class carrying chrome drawings.

        Page totals are exact in a single build: at save() time every
        page has been recorded, so len(self._saved) is the true total.
        multiBuild (needed for the TOC) re-runs the story; each pass
        paints with its own correct total.
        """
        from reportlab.pdfgen.canvas import Canvas as _Canvas

        class _Numbered(_Canvas):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._saved = []
                self._page_no = 0

            def showPage(self):
                self._saved.append((dict(self.__dict__), self._page_no + 1))
                self._page_no += 1
                self._startPage()

            def save(self):
                total = len(self._saved)
                for saved, number in self._saved:
                    self.__dict__.update(saved)
                    cls._paint(
                        self, number, total,
                        header_left, header_right, footer_left, footer_mid,
                    )
                    super().showPage()
                super().save()

        return _Numbered

    @staticmethod
    def _paint(canvas, number, total, header_left, header_right,
               footer_left, footer_mid):
        if number == 1:
            return  # cover page stays clean
        canvas.saveState()
        canvas.setFont(BOLD_FONT, 7.5)
        canvas.setFillColor(NAVY)
        canvas.drawString(MARGIN, A4[1] - 14 * mm, header_left)
        canvas.setFont(BODY_FONT, 7.5)
        canvas.drawRightString(A4[0] - MARGIN, A4[1] - 14 * mm, header_right)
        canvas.setStrokeColor(GRAY_LINE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, A4[1] - 16.5 * mm, A4[0] - MARGIN, A4[1] - 16.5 * mm)

        canvas.setFont(BODY_FONT, 7.5)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        footer = f"{footer_left}    |    {footer_mid}    |    Page {number} of {total}"
        canvas.drawCentredString(A4[0] / 2, 12 * mm, footer)
        canvas.restoreState()


def deepcopy_styles(styles):
    return deepcopy(styles)
