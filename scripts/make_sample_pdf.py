"""Generate sample_docs/finance/expense-policy.pdf (3 pages) so PDF page citations can be tested.

Usage: python scripts/make_sample_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
    ("Expense Policy - Overview",
     ["This policy explains which business expenses Northwind Labs reimburses and how to claim them.",
      "Expenses must be submitted in the expense tool within 30 days of the purchase date. "
      "Claims submitted after 30 days need written approval from the finance controller.",
      "Every expense of 25 US dollars or more needs an itemized receipt."]),
    ("Expense Policy - Approvals",
     ["Expenses up to 1,000 US dollars are approved by your direct manager.",
      "Expenses above 1,000 US dollars also need approval from the department head.",
      "Software subscriptions of any amount must be approved by the IT team before purchase, "
      "so that security can review the vendor."]),
    ("Expense Policy - Reimbursement",
     ["Approved expenses are reimbursed through payroll on the next pay date, usually within 10 business days.",
      "Corporate card holders must reconcile their card statement by the 5th of each month.",
      "Personal items, traffic fines and gifts above 50 US dollars are not reimbursable."]),
]


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "sample_docs" / "finance" / "expense-policy.pdf"
    styles = getSampleStyleSheet()
    story = []
    for i, (heading, paras) in enumerate(PAGES):
        story.append(Paragraph(heading, styles["Heading1"]))
        for p in paras:
            story += [Paragraph(p, styles["BodyText"]), Spacer(1, 8)]
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(out), pagesize=letter, title="Expense Policy", invariant=1).build(story)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
