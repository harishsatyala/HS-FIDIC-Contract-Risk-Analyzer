"""Domain knowledge: FIDIC-language lexicon for the 30 risk-library categories, onerous-term
triggers, and a mitigation playbook.

The texts below are original guidance written for this tool (not reproduced FIDIC wording).
Sub-Clause references point to the FIDIC 2017 Red/Yellow Book numbering for orientation only.
"""
from __future__ import annotations

# category -> (lexicon regex fragments, base impact 1-5, FIDIC orientation, exposure factor of contract value,
#              typical mitigation effectiveness, playbook mitigation)
CATEGORIES: dict[str, dict] = {
    "Contractual": dict(
        terms=[r"ambigu", r"discrepanc", r"priority of documents", r"inconsisten", r"conflict(?:ing)? (?:between|with)",
               r"interpretation", r"entire agreement", r"governing law", r"\bamend", r"particular conditions", r"precedence"],
        impact=3, ref="SC 1.5 Priority of Documents", exposure=0.02, eff=0.6,
        mitigation="Request clarification during tender; insist on a clear order-of-precedence clause; log every discrepancy as a tender qualification."),
    "Schedule": dict(
        terms=[r"time for completion", r"extension of time", r"delay", r"programme", r"schedule", r"milestone",
               r"commencement date", r"rate of progress", r"accelerat", r"critical path", r"suspension"],
        impact=4, ref="SC 8.2 / 8.5 Time for Completion & EoT", exposure=0.05, eff=0.5,
        mitigation="Build float into the baseline programme; secure EoT entitlement for Employer-caused delay; keep contemporaneous delay records; issue early-warning notices."),
    "Cost & Estimating": dict(
        terms=[r"contract price", r"lump sum", r"accepted contract amount", r"bill of quantities", r"escalat",
               r"adjustment for changes in cost", r"rates? and prices?", r"\bestimat", r"quantit(?:y|ies)", r"fixed price"],
        impact=4, ref="SC 13.7 / 14.1 Contract Price & cost adjustment", exposure=0.05, eff=0.45,
        mitigation="Price escalation through an indexation formula; carry explicit contingency; qualify quantities that are the Employer's responsibility."),
    "Design & Engineering": dict(
        terms=[r"design", r"fit for (?:the )?purpose", r"employer'?s requirements", r"drawings", r"specification",
               r"contractor'?s documents", r"errors? in", r"engineering", r"as-built"],
        impact=4, ref="SC 4.1 / 5.1 Design obligations", exposure=0.04, eff=0.55,
        mitigation="Limit fitness-for-purpose to defined performance criteria; verify Employer's Requirements during tender; obtain design PI insurance; formal design reviews."),
    "Quality & Defects": dict(
        terms=[r"defect", r"remedy", r"quality", r"inspection", r"testing", r"tests? on completion", r"reject",
               r"nonconform", r"workmanship", r"defects notification period", r"performance certificate"],
        impact=3, ref="SC 7 / 11 Plant, Materials, Defects", exposure=0.03, eff=0.6,
        mitigation="Robust QA/QC plan and ITPs; back-to-back defect warranties from suppliers; cap Defects Notification Period extensions."),
    "HSE": dict(
        terms=[r"health", r"safety", r"accident", r"injur", r"\bhse\b", r"hazard", r"protection of the environment",
               r"safety procedures", r"welfare"],
        impact=4, ref="SC 4.8 / 6.7 Health and Safety", exposure=0.02, eff=0.65,
        mitigation="Site-specific HSE plan; competent HSE staff; permit-to-work system; incident reporting and lessons learned."),
    "Environmental": dict(
        terms=[r"environment", r"pollut", r"contaminat", r"emission", r"waste", r"discharge", r"noise", r"hazardous material"],
        impact=3, ref="SC 4.18 Protection of the Environment", exposure=0.02, eff=0.55,
        mitigation="Baseline environmental survey; clear allocation of pre-existing contamination to the Employer; waste-management plan."),
    "Political & Country": dict(
        terms=[r"\bwar\b", r"hostilit", r"riot", r"civil (?:war|disturbance|commotion)", r"terroris", r"expropriat",
               r"sanction", r"embargo", r"currenc(?:y|ies) (?:restriction|transfer)", r"country"],
        impact=4, ref="SC 18 Exceptional Events", exposure=0.04, eff=0.35,
        mitigation="Political-risk insurance; currency convertibility provisions; Employer-risk allocation of country events."),
    "Regulatory & Change in Law": dict(
        terms=[r"change(?:s)? in (?:the )?laws?", r"changes? in legislation", r"\blaws?\b", r"regulation", r"statut",
               r"legislation", r"decree", r"tax(?:es|ation)?", r"duties"],
        impact=3, ref="SC 13.6 Adjustments for Changes in Laws", exposure=0.02, eff=0.5,
        mitigation="Secure a change-in-law adjustment clause with a clear base date; monitor legislative pipeline; tax advice for local duties."),
    "Permits & Approvals": dict(
        terms=[r"permit", r"licen[cs]e", r"approval", r"consent", r"authori[sz]ation", r"public authorit"],
        impact=3, ref="SC 1.13 Compliance with Laws", exposure=0.02, eff=0.55,
        mitigation="Split permit responsibilities in a permits matrix; Employer to obtain planning/zoning consents; EoT for authority delays."),
    "Site & Ground Conditions": dict(
        terms=[r"site data", r"physical conditions", r"sub-?surface", r"ground", r"geolog", r"hydrolog", r"unforeseeable",
               r"utilit(?:y|ies)", r"soil", r"rock", r"archaeolog", r"fossil"],
        impact=4, ref="SC 4.12 Unforeseeable Physical Conditions", exposure=0.05, eff=0.45,
        mitigation="Independent site investigation before bid; reject clauses deeming the Contractor to have verified all site data; geotechnical baseline report."),
    "Weather & Natural Hazards": dict(
        terms=[r"weather", r"climat", r"flood", r"earthquake", r"storm", r"hurricane", r"typhoon", r"natural catastroph",
               r"rain", r"volcan", r"tsunami"],
        impact=3, ref="SC 8.5 (exceptionally adverse climatic conditions)", exposure=0.02, eff=0.45,
        mitigation="Weather-day allowance in programme; CAR insurance with natural-peril cover; define 'exceptionally adverse' objectively."),
    "Force Majeure / Exceptional Events": dict(
        terms=[r"force majeure", r"exceptional event", r"act of god", r"epidemic", r"pandemic", r"beyond (?:its|the party'?s) control"],
        impact=4, ref="SC 18 Exceptional Events", exposure=0.03, eff=0.4,
        mitigation="Ensure Exceptional Events give time and cost relief; include termination right after prolonged prevention; business-interruption cover."),
    "Procurement & Supply Chain": dict(
        terms=[r"subcontract", r"supplier", r"procure", r"nominated", r"vendor", r"long[- ]lead", r"purchase", r"manufactur"],
        impact=3, ref="SC 4.4 / 5 Subcontractors & Nominated Subcontractors", exposure=0.03, eff=0.55,
        mitigation="Back-to-back subcontract terms; dual sourcing for long-lead items; right to object to nominated subcontractors."),
    "Logistics & Customs": dict(
        terms=[r"import", r"export", r"customs", r"transport", r"shipping", r"port", r"freight", r"delivery", r"route"],
        impact=3, ref="SC 4.16 Transport of Goods", exposure=0.02, eff=0.5,
        mitigation="Incoterms clarity; customs-clearance responsibility with the Employer where duty exemptions apply; route surveys for abnormal loads."),
    "Labour & Skills": dict(
        terms=[r"labour", r"labor", r"staff", r"personnel", r"workforce", r"recruit", r"key personnel", r"working hours", r"wages"],
        impact=2, ref="SC 6 Staff and Labour", exposure=0.015, eff=0.6,
        mitigation="Early mobilisation and retention plan; local-labour availability study; compliance with labour laws and camp standards."),
    "Plant & Equipment": dict(
        terms=[r"contractor'?s equipment", r"\bplant\b", r"machinery", r"equipment", r"spare parts", r"temporary works"],
        impact=3, ref="SC 4.17 / 7 Contractor's Equipment, Plant", exposure=0.02, eff=0.55,
        mitigation="Equipment maintenance regime; standby plant; equipment-breakdown insurance."),
    "Interfaces": dict(
        terms=[r"interface", r"other contractors", r"co-?operat", r"coordinat", r"access route", r"by others"],
        impact=3, ref="SC 4.6 Co-operation", exposure=0.03, eff=0.45,
        mitigation="Interface responsibility matrix; Employer coordination obligation; compensation for disruption by other contractors."),
    "Stakeholder & Community": dict(
        terms=[r"communit", r"stakeholder", r"public", r"neighbour", r"third part(?:y|ies)", r"interference", r"protest", r"nuisance"],
        impact=2, ref="SC 4.14 Avoidance of Interference", exposure=0.015, eff=0.5,
        mitigation="Stakeholder engagement plan; Employer responsibility for social licence; grievance mechanism."),
    "Land & Right of Way": dict(
        terms=[r"right of access", r"possession of (?:the )?site", r"access to(?: the)? site", r"wayleave", r"easement",
               r"land acquisition", r"right of way", r"possession of (?:all )?(?:parts of )?the site"],
        impact=4, ref="SC 2.1 Right of Access to the Site", exposure=0.04, eff=0.45,
        mitigation="Fix site-access dates in the contract; EoT + cost for late access; phase works to match land availability."),
    "Payment & Cash Flow": dict(
        terms=[r"payment", r"interim payment", r"advance payment", r"retention", r"invoice", r"certif", r"financing charges",
               r"withh[eo]ld", r"set[- ]off", r"deduct", r"financial arrangements", r"currenc"],
        impact=4, ref="SC 14 Contract Price and Payment", exposure=0.04, eff=0.55,
        mitigation="Positive cash-flow payment schedule; advance payment against guarantee; financing charges for late payment; right to suspend for non-payment."),
    "Claims & Disputes": dict(
        terms=[r"claim", r"dispute", r"arbitrat", r"adjudicat", r"\bdab\b", r"\bdaab\b", r"time[- ]bar", r"notice of dissatisfaction",
               r"amicable settlement", r"determination", r"claims? (?:for|procedure)", r"discharged from (?:all )?liabilit", r"give (?:a )?notice"],
        impact=3, ref="SC 20 / 21 Claims, Disputes and Arbitration", exposure=0.03, eff=0.55,
        mitigation="Contract-administration system tracking notice periods; claims register; avoid unduly short time-bars; standing DAAB."),
    "Change & Scope": dict(
        terms=[r"variation", r"\bchange", r"omission", r"instruct", r"scope", r"additional work", r"value engineering", r"adjust"],
        impact=3, ref="SC 13 Variations and Adjustments", exposure=0.03, eff=0.55,
        mitigation="Valuation rules for variations; cap on omissions without compensation; written-instruction requirement before executing changes."),
    "Technology & Maturity": dict(
        terms=[r"technolog", r"innovat", r"prototype", r"proven", r"patent", r"intellectual property", r"software", r"novel"],
        impact=3, ref="SC 17.3 Intellectual and Industrial Property", exposure=0.03, eff=0.45,
        mitigation="Use proven technology or pilot tests; IP indemnity from technology suppliers; performance guarantees tied to tested parameters."),
    "Digital, Data & Cybersecurity": dict(
        terms=[r"cyber", r"(?<!contract )(?<!site )\bdata\b", r"electronic", r"\bbim\b", r"information system", r"confidential", r"digital"],
        impact=2, ref="SC 1.12 Confidentiality / 1.3 Communications", exposure=0.01, eff=0.6,
        mitigation="Data-handling protocol; cybersecurity requirements for project systems; confidentiality carve-outs."),
    "Commissioning & Start-up": dict(
        terms=[r"commission", r"tests after completion", r"start[- ]up", r"taking[- ]over", r"trial operation", r"performance test", r"handover"],
        impact=4, ref="SC 9 / 10 Tests on Completion & Taking Over", exposure=0.04, eff=0.5,
        mitigation="Define test procedures and acceptance criteria in advance; Employer to provide feedstock/utilities for tests; deemed taking-over if tests are prevented."),
    "Operations & Maintainability": dict(
        terms=[r"operation and maintenance", r"maintenance", r"o&m", r"operating manuals", r"training", r"spares", r"availability"],
        impact=2, ref="SC 5.5 / 5.7 Training and O&M manuals", exposure=0.015, eff=0.6,
        mitigation="Agree O&M deliverables and training scope; spares list with pricing; availability guarantees with clear exclusions."),
    "Governance & Decision Making": dict(
        terms=[r"engineer'?s determination", r"\bengineer\b", r"representative", r"delegat", r"authority", r"decision",
               r"sole discretion", r"opinion of the", r"satisfaction of the"],
        impact=2, ref="SC 3 The Engineer / Employer's Administration", exposure=0.015, eff=0.5,
        mitigation="Require objective criteria and response times for approvals; deemed-approval provisions; escalation ladder."),
    "Fraud, Ethics & Compliance": dict(
        terms=[r"corrupt", r"brib", r"fraud", r"collusi", r"conflict of interest", r"anti-?corruption", r"sanction", r"integrity", r"inducement"],
        impact=3, ref="SC 15.6 Corrupt or Fraudulent Practices", exposure=0.02, eff=0.7,
        mitigation="Compliance programme and due diligence on agents/subcontractors; audit rights; training."),
    "Insurance & Security": dict(
        terms=[r"insur", r"performance security", r"bond", r"guarantee", r"indemnif", r"liabilit", r"limitation of liability",
               r"security", r"theft", r"vandal", r"care of the works"],
        impact=4, ref="SC 4.2 / 17 / 19 Security, Indemnities, Insurance", exposure=0.04, eff=0.55,
        mitigation="Cap aggregate liability (e.g. at Accepted Contract Amount) and exclude indirect loss; align bond amounts and expiry; check insurance gaps and deductibles."),
}

# Onerous / risk-shifting triggers: (regex, label, severity 1-5, likelihood weight 0-1, specific mitigation)
TRIGGERS = [
    (r"liquidated damages|delay damages|\bpenalt(?:y|ies)\b", "Delay damages / penalties", 4, 0.55,
     "Negotiate a cap on delay damages (typically 5–10% of contract value) and make them the sole remedy for delay."),
    (r"unlimited liabilit|without limit(?:ation)?|shall not be limited", "Uncapped liability", 5, 0.65,
     "Insert an aggregate liability cap and carve out only fraud / wilful misconduct."),
    (r"indemnif(?:y|ies)|hold harmless", "Indemnity obligation", 4, 0.45,
     "Make indemnities mutual and limited to losses caused by the indemnifying party's negligence or breach."),
    (r"at (?:the contractor'?s|its|his|their) (?:own )?(?:sole )?(?:cost|risk|expense)", "At own cost / risk", 4, 0.5,
     "Price the item explicitly or request that the cost risk is shared or reimbursable."),
    (r"sole (?:and absolute )?discretion|absolute discretion|in (?:its|his) opinion", "Unilateral discretion", 3, 0.4,
     "Replace discretion with objective criteria and a 'not to be unreasonably withheld or delayed' standard."),
    (r"time[- ]barr|shall not be entitled|no (?:claim|entitlement)|shall have no (?:right|claim|liability)|waive", "Entitlement bar / waiver", 4, 0.5,
     "Monitor notice periods with a claims register; negotiate longer notice periods and remove waiver wording."),
    (r"deemed to have|shall be deemed|is deemed", "Deeming provision", 3, 0.35,
     "Verify deemed facts (site data, sufficiency of price) during tender; qualify the bid where verification is impossible."),
    (r"fit for (?:the )?purpose|fitness for purpose", "Fitness-for-purpose", 4, 0.45,
     "Define the purpose objectively in the Employer's Requirements; align with design-insurance cover (reasonable skill and care)."),
    (r"terminat", "Termination", 4, 0.35,
     "Ensure termination for convenience pays demobilisation, committed costs and loss of profit; add cure periods before default termination."),
    (r"suspen", "Suspension", 3, 0.3,
     "Secure time and cost for suspension; right to terminate after prolonged suspension."),
    (r"withh[eo]ld|set[- ]off|deduct", "Withholding / set-off", 3, 0.4,
     "Restrict set-off to certified or agreed amounts; require notice before deduction."),
    (r"unforeseeable|unforeseen", "Unforeseeable conditions", 4, 0.4,
     "Retain the unforeseeable-conditions relief; define the baseline (site data) against which foreseeability is judged."),
    (r"\bforce majeure\b|exceptional event", "Exceptional event", 3, 0.3,
     "Confirm time + cost relief for Employer-risk events and a termination right after prolonged prevention."),
    (r"no (?:extension|additional payment|adjustment)|shall not be adjusted|fixed and not subject", "No adjustment / fixed", 4, 0.5,
     "Qualify the price or add escalation / change-in-law relief."),
    (r"consequential|indirect loss|loss of profit", "Consequential loss", 4, 0.4,
     "Mutual exclusion of indirect and consequential loss, including loss of profit and production."),
    (r"\brisks?\b|\bhazard", "Explicit risk allocation", 3, 0.35,
     "Confirm who bears the risk and price or insure the residual exposure."),
    (r"\bliab(?:le|ility)\b", "Liability", 3, 0.3,
     "Check the liability cap, exclusions and insurance alignment."),
    (r"\bshall\b.*\bwithin \d+ (?:days|hours)|\bwithin \d+ days\b", "Short time limit", 2, 0.25,
     "Track the time limit in the contract-administration calendar; negotiate a realistic period."),
]

PROB_LEVEL = {"L": 0.25, "M": 0.5, "H": 0.75}
