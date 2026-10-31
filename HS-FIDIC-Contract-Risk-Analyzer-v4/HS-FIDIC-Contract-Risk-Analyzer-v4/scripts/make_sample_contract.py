"""Generate an original, FIDIC-style sample EPC contract PDF for demos and tests.
The wording is written for this project and is not reproduced from FIDIC publications."""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

CLAUSES = [
    ("1", "General Provisions", [
        ("1.1", "Definitions", [
            "“Contract Price” means the lump sum stated in the Contract Agreement, subject to adjustment only as expressly provided in this Contract.",
            "“Site” means the land on which the Permanent Works are to be executed, together with any other places designated in the Contract as forming part of the Site.",
        ]),
        ("1.5", "Priority of Documents", [
            "If an ambiguity or discrepancy is found between the documents, the Engineer shall issue any necessary clarification and the Contractor shall not be entitled to any additional payment as a result.",
        ]),
    ]),
    ("2", "The Employer", [
        ("2.1", "Right of Access to the Site", [
            "The Employer shall give the Contractor right of access to and possession of all parts of the Site within 60 days after the Commencement Date.",
            "If the Contractor suffers delay as a result of a failure by the Employer to give access to the Site, the Contractor shall be entitled to an extension of time but shall not be entitled to any additional cost.",
        ]),
        ("2.4", "Employer's Financial Arrangements", [
            "The Employer may withhold any payment certified by the Engineer if, in its sole discretion, the Contractor has failed to perform any obligation under the Contract.",
        ]),
    ]),
    ("4", "The Contractor", [
        ("4.1", "Contractor's General Obligations", [
            "The Contractor shall design, execute and complete the Works in accordance with the Contract and the Works shall be fit for the purposes for which they are intended.",
            "The Contractor shall provide all Contractor's Equipment, labour and materials required for the execution of the Works.",
        ]),
        ("4.2", "Performance Security", [
            "The Contractor shall obtain a performance security in the amount of 15% of the Contract Price at its own cost and shall keep it valid until the issue of the Performance Certificate.",
        ]),
        ("4.10", "Use of Site Data", [
            "The Contractor shall be deemed to have inspected and examined the Site and its surroundings and to have satisfied itself as to the sub-surface, hydrological and climatic conditions before submitting the Tender.",
            "The Employer shall have no liability for the accuracy or completeness of any site data supplied by or on behalf of the Employer.",
        ]),
        ("4.12", "Unforeseeable Physical Conditions", [
            "The Contractor shall bear all risks of unforeseeable physical conditions, including ground conditions, utilities and contamination encountered on the Site, at its own cost.",
        ]),
        ("4.18", "Protection of the Environment", [
            "The Contractor shall take all necessary measures to protect the environment and shall indemnify and hold harmless the Employer against all claims arising from pollution caused by the Works.",
        ]),
    ]),
    ("8", "Commencement, Delays and Suspension", [
        ("8.2", "Time for Completion", [
            "The Contractor shall complete the whole of the Works within the Time for Completion stated in the Contract Data.",
        ]),
        ("8.8", "Delay Damages", [
            "If the Contractor fails to comply with the Time for Completion, the Contractor shall pay liquidated damages to the Employer of 0.5% of the Contract Price for every day of delay, and such damages shall not be limited.",
        ]),
        ("8.9", "Employer's Suspension", [
            "The Engineer may at any time instruct the Contractor to suspend progress of part or all of the Works, and the Contractor shall protect and secure the suspended Works at its own cost.",
        ]),
    ]),
    ("13", "Variations and Adjustments", [
        ("13.1", "Right to Vary", [
            "Variations may be initiated by the Engineer at any time prior to issuing the Taking-Over Certificate by an instruction.",
            "The Contractor shall not be entitled to any adjustment of the Contract Price for omissions of work of up to 20% of the Contract Price.",
        ]),
        ("13.6", "Adjustments for Changes in Laws", [
            "The Contract Price shall be fixed and not subject to adjustment for any change in the Laws of the Country, including changes in taxes and duties.",
        ]),
    ]),
    ("14", "Contract Price and Payment", [
        ("14.3", "Application for Interim Payments", [
            "The Contractor shall submit a Statement to the Engineer after the end of each month in the form approved by the Engineer.",
        ]),
        ("14.7", "Payment", [
            "The Employer shall pay to the Contractor the amount certified in each Interim Payment Certificate within 90 days after the Engineer receives the Statement.",
            "The Employer shall be entitled to deduct from any payment any amount which, in its opinion, is due from the Contractor.",
        ]),
    ]),
    ("17", "Care of the Works and Indemnities", [
        ("17.1", "Responsibility for Care of the Works", [
            "The Contractor shall take full responsibility for the care of the Works from the Commencement Date until the Taking-Over Certificate is issued, and any loss or damage shall be rectified at the Contractor's risk and cost.",
        ]),
        ("17.6", "Limitation of Liability", [
            "Neither Party shall be liable to the other Party for loss of use of any Works, loss of profit or any indirect or consequential loss, except for the Contractor's liability under Sub-Clause 8.8.",
        ]),
    ]),
    ("18", "Exceptional Events", [
        ("18.1", "Exceptional Events", [
            "Exceptional Event means an event which is beyond a Party's control, including war, riot, natural catastrophes such as earthquake, flood and hurricane, and epidemic.",
            "If the Contractor is prevented from performing by an Exceptional Event, the Contractor shall be entitled to an extension of time but shall not be entitled to payment of any Cost.",
        ]),
    ]),
    ("19", "Insurance", [
        ("19.1", "General Requirements", [
            "The Contractor shall effect and maintain insurances for the Works, Contractor's Equipment and third-party liability at its own expense, with deductibles borne by the Contractor.",
        ]),
    ]),
    ("20", "Claims", [
        ("20.2", "Claims for Payment and Extension of Time", [
            "If the Contractor considers itself entitled to any extension of time or additional payment, the Contractor shall give notice to the Engineer within 14 days after becoming aware of the event, otherwise the Contractor shall not be entitled to any claim and the Employer shall be discharged from all liability.",
        ]),
    ]),
    ("21", "Disputes and Arbitration", [
        ("21.6", "Arbitration", [
            "Any dispute which is not amicably settled shall be finally settled by international arbitration under the Rules of Arbitration of the International Chamber of Commerce.",
        ]),
    ]),
]


def main(out="data/sample_epc_contract.pdf"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out, pagesize=A4, title="Sample EPC Contract (synthetic)")
    story = [Paragraph("SAMPLE EPC CONTRACT – GENERAL CONDITIONS (synthetic demonstration document)", styles["Title"]),
             Paragraph("Solar PV Plant, 120 MW – illustrative clauses written for HS-FIDIC testing; not FIDIC text.", styles["Italic"]),
             Spacer(1, 12)]
    for no, title, subs in CLAUSES:
        story.append(Paragraph(f"{no} {title}", styles["Heading2"]))
        for sno, stitle, paras in subs:
            story.append(Paragraph(f"{sno} {stitle}", styles["Heading4"]))
            for p in paras:
                story.append(Paragraph(p, styles["BodyText"]))
                story.append(Spacer(1, 4))
    Path(out).parent.mkdir(exist_ok=True, parents=True)
    doc.build(story)
    print("wrote", out)


if __name__ == "__main__":
    main()
