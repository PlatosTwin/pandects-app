"""
Measure how much of the corpus the taxonomy packs cover, by section title.

This is a coverage check for docs/plans/full-taxonomy.md. It is not a labeler:
labels will come from body text (see the plan). It reads a TSV exported with a
read-only query and writes nothing but a report to stdout.

Export the input from the repo root (read-only):
  ./bin/mariadb-read "SELECT LEFT(agreement_uuid,8) a, deal_type, target_type, \
      article_title, section_title, section_standard_ids \
      FROM latest_sections_search" > sections.tsv

Run:
  etl/.venv/bin/python docs/plans/full-taxonomy/title_coverage.py sections.tsv \
      [--unmapped 200] [--by-concept]

Each title is split on ";" into parts. A rule maps a part to one concept (a
leaf) or to a navigation node (a section that normally holds several leaves).
A section is:
  leaf       every part maps to a specific concept;
  group      every part maps, and at least one maps to a navigation node;
  partial    some parts map and some do not;
  catch_all  the title maps only to a catch-all concept (reserved, other);
  untitled   the section has no title;
  unmapped   no part maps.
"""

from __future__ import annotations

import argparse
import collections
import csv
import random
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "etl" / "src"))

csv.field_size_limit(10**9)

# Article contexts, read from the article title.
CONTEXTS = {
    "REP": r"representation|warrant",
    "COND": r"condition",
    "TERM": r"terminat",
    "INDEM": r"indemni|survival|remedies",
    "TAX": r"\btax",
    "DEF": r"definition|defined terms|interpretation",
    "MISC": r"miscellaneous|general|other provisions",
    "CONSID": r"consideration|purchase price|conversion|exchange of|payment|effect of the merger on|effect on capital|the merger|purchase and sale|the closing|the offer",
    "EMP": r"employee|benefit",
    "COV": r"covenant|agreements|conduct|pending",
}

# (context or None, regex on one title part, target id). First match wins.
# Context rules come first so that a short title such as "insurance" resolves by its article.
RULES: list[tuple[str | None, str, str]] = [
    # ---- placeholders
    (None, r"^\[?(intentionally (omitted|deleted|left blank)|reserved|omitted|deleted)\]?\.?$|^\[reserved", "ma.other.reserved"),
    # ---- specific phrases that mean the same thing under any article
    (None, r"rule 145|affiliate letters?|^affiliates$|affiliate agreements$|pooling", "ma.rule_145_affiliates"),
    (None, r"title (insurance|commitments?|defects?|matters|polic|and survey|examination|review)|surveys?( matters)?$|environmental (defects?|assessment|review|examination)|defect (notice|procedures)", "ma.asset.title_and_survey"),
    (None, r"systems? conversion|operating functions|integration|data processing|conversion (planning|and transition)|transition planning|informational systems", "ma.conduct.integration_planning"),
    (None, r"reservation of right|right to revise", "ma.structure.alternative_structure"),
    (None, r"reservation of|reserve for issuance", "ma.consideration.buyer_securities"),
    (None, r"long.?stop date", "ma.termination.outside_date"),
    (None, r"^no trading|trading in (acquiror|spac|parent) securities", "ma.covenant.other"),
    (None, r"^business activities$|business activities of|activities of (acquiror|spac|purchaser|parent)$", "ma.spac.business_activities"),
    (None, r"restrictions? on business", "ma.rep.material_contracts"),
    (None, r"jobs act|emerging growth", "ma.rep.investment_company"),
    (None, r"accuracy of representations|representations (and warranties )?(true|accurate)", "ma.condition.reps_bringdown"),
    (None, r"schedule to$", "ma.tender.offer"),
    # ---- representations (context)
    ("REP", r"^(organization|incorporation|existence|qualification|good standing|due (organization|incorporation)|corporate (organization|existence|status)|organizational documents|governing documents|charter|certificate of incorporation|articles of incorporation|bylaws|standing|power)\b(?!.*authori[tz])", "ma.rep.organization"),
    ("REP", r"organi[sz]ation.*(standing|qualification|power|existence)|existence and (power|good standing)|qualification, organization", "ma.rep.organization"),
    ("REP", r"vote required|required vote|stockholder approval|shareholder approval|voting requirements|requisite (vote|approval)|vote/approval", "ma.rep.required_vote"),
    ("REP", r"authori[tz]|enforceab|binding (effect|agreement|obligation|nature)|execution and delivery|power and authority|board (approval|recommendation|action)|validity of agreement|corporate (power|action)", "ma.rep.authority"),
    ("REP", r"(title|ownership) (to|of) (the )?(shares|stock|interests|units|securities|membership|equity|purchased (shares|interests))|ownership of (company )?(shares|capital stock|interests)$", "ma.rep.title_to_shares"),
    ("REP", r"ownership of (company|target|parent|seller) (common )?(stock|shares|securities)|ownership of shares|interested (stockholder|shareholder)|share ownership|no ownership", "ma.rep.ownership_of_target_stock"),
    ("REP", r"subsidiar|equity (investments|interests)", "ma.rep.subsidiaries"),
    ("REP", r"capitali[sz]ation|capital (stock|structure)|authorized (capital|shares)|outstanding (shares|securities)", "ma.rep.capitalization"),
    ("REP", r"valid issuance|issuance of (shares|securities|parent)|shares to be issued|parent (common )?(stock|shares)$", "ma.rep.valid_issuance"),
    ("REP", r"conflict|contravention|violation|consents|approvals|governmental (authori[sz]ation|filings|consent)|required (filings|consents)|filings|no (breach|default)|regulatory approvals", "ma.rep.no_conflict"),
    ("REP", r"sec (reports|filings|documents)|reports and financial|securities (filings|laws)|exchange act|public (filings|reports)|sarbanes|regulatory reports|^reports$", "ma.rep.sec_reports"),
    ("REP", r"internal (accounting )?controls|disclosure controls", "ma.rep.internal_controls"),
    ("REP", r"undisclosed|^(no |absence of )?(material )?liabilities|absence of liabilities", "ma.rep.undisclosed_liabilities"),
    ("REP", r"financial (statements|information|reports)|^financials", "ma.rep.financial_statements"),
    ("REP", r"books and records|minute books|corporate records|bank accounts|powers of attorney|^records$", "ma.rep.books_records"),
    ("REP", r"absence of (certain )?(changes|developments|events)|no (material adverse (change|effect)|changes|adverse change)|subsequent events|events subsequent|conduct of business|certain (changes|developments)|material adverse (change|effect)|ordinary course", "ma.rep.absence_of_changes"),
    ("REP", r"receivable|inventor|payables", "ma.rep.receivables_inventory"),
    ("REP", r"litigation|proceedings|^actions|claims|orders|investigations|legal matters", "ma.rep.litigation"),
    ("REP", r"corrupt|bribery|sanction|export|trade (controls|laws)|business practices|improper payments|unlawful payments|money laundering|ofac|certain payments|international trade", "ma.rep.anti_corruption_sanctions"),
    ("REP", r"investment company|1940 act|jobs act|emerging growth", "ma.rep.investment_company"),
    ("REP", r"health ?care|fda|regulatory (matters|compliance)|medicare|product registrations|clinical", "ma.rep.healthcare_regulatory"),
    ("REP", r"loan|deposits|allowance|community reinvestment|fiduciary|trust business|regulatory agenc|derivative|risk management|investment securities|bank (secrecy|holding)|interest rate|mortgage|supervisory|cra\b|fdic|capital adequacy|well.capitali[sz]ed", "ma.rep.banking"),
    ("REP", r"reserves|reinsurance|insurance (business|subsidiar|contracts|regulat|matters)|statutory (statements|financial)|actuarial|producers", "ma.rep.insurance_business"),
    ("REP", r"oil and gas|hydrocarbon|wells|reserve report|mineral|mining|royalt|utility|energy regul|ferc|puhca|pipeline", "ma.rep.energy_natural_resources"),
    ("REP", r"compliance|permits|licenses|legal requirements|governmental authorizations|applicable law", "ma.rep.compliance_permits"),
    ("REP", r"government(al)? contract", "ma.rep.government_contracts"),
    ("REP", r"customers|suppliers|vendors|distributors", "ma.rep.customers_suppliers"),
    ("REP", r"affiliate|related (party|person)|interested (party|person)|insider|certain (relationships|interests|transactions)|interests of (officers|directors)", "ma.rep.affiliate_transactions"),
    ("REP", r"product|^warrant(y|ies)$|recalls", "ma.rep.product_liability_warranty"),
    ("REP", r"contracts|agreements|commitments|leases and contracts", "ma.rep.material_contracts"),
    ("REP", r"insurance", "ma.rep.insurance"),
    ("REP", r"real (property|estate)|leases|leased|premises|properties$|^property$", "ma.rep.real_property"),
    ("REP", r"title|sufficiency|assets|personal property|properties|equipment|condition of", "ma.rep.title_sufficiency_assets"),
    ("REP", r"privacy|data|cyber|information technology|it systems|computer systems|security", "ma.rep.privacy_cybersecurity"),
    ("REP", r"intellectual|proprietary|patents|trademarks|software|technology", "ma.rep.intellectual_property"),
    ("REP", r"environment|hazardous", "ma.rep.environmental"),
    ("REP", r"labor|employment (matters|practices|relations)|employees(?!.*(plan|benefit))|employee (matters|relations)$|collective bargaining|unions|personnel|workers", "ma.rep.labor_employment"),
    ("REP", r"benefit|erisa|plans|pension|compensation|employee", "ma.rep.employee_benefits"),
    ("REP", r"^tax|taxes|taxation|tax (matters|returns|status|treatment)|reorgani[sz]ation|tax-free|firpta", "tax.nav.reps"),
    ("REP", r"information (supplied|provided|statement)|proxy|registration statement|disclosure documents|offer documents|schedule 14d|form s-4|information in|company information|parent information", "ma.rep.information_supplied"),
    ("REP", r"opinion|fairness|financial advisor", "ma.rep.fairness_opinion"),
    ("REP", r"broker|finder|advisory fees|financial advisors", "ma.rep.brokers"),
    ("REP", r"(full |accuracy of )?disclosure$|no (misstatements|misleading)|material (misstatements|facts)|complete(ness)? (of|and)|accuracy of (information|statements)|10b-5|untrue statements", "ma.rep.full_disclosure"),
    ("REP", r"solven|fraudulent (transfer|conveyance)", "ma.rep.solvency"),
    ("REP", r"financing|funds|financial (capability|capacity|ability|resources)|commitment letter|capital resources", "ma.rep.financing"),
    ("REP", r"investment (intent|purpose|representation)|purchase for investment|accredited|securities act|investor|acquisition for investment|investment$|restricted securities|purchase entirely for own account", "ma.rep.investment_intent"),
    ("REP", r"merger sub|acquisition sub|operations of|no prior activities|interim operations of|formation of|activities of", "ma.rep.merger_sub"),
    ("REP", r"takeover|anti-?takeover|state takeover|business combination statute|section 203", "ma.takeover.statutes"),
    ("REP", r"rights (agreement|plan)", "ma.takeover.rights_plan"),
    ("REP", r"no other|no additional|disclaimer|exclusiv|no implied|limitation(s)? (of|on) (representations|warranties)|as.is", "ma.rep.no_other_reps"),
    ("REP", r"reliance|independent (investigation|review)|investigation|due diligence|acknowledg|inspection", "ma.rep.non_reliance"),
    ("REP", r"trust (account|fund)", "ma.spac.trust_account"),
    ("REP", r"pipe|subscription agreement", "ma.spac.pipe"),
    ("REP", r"guarant", "ma.financing.limited_guarantee"),
    ("REP", r"knowledge", "ma.def.knowledge"),
    ("REP", r"representations and warranties|^general$|^representations$|schedules|^survival", "ma.nav.reps"),
    ("REP", r"indebtedness|^debt", "ma.rep.other"),
    # ---- conditions (context)
    ("COND", r"each party|all parties|both parties|mutual|the parties|obligations of (the )?(company|seller|sellers|target) and|each of the parties|to the merger$|^conditions to (the )?(closing|merger)$|precedent to (the )?(obligations of )?(each|all|both)", "ma.nav.conditions.mutual"),
    ("COND", r"conditions|obligation", "ma.nav.conditions.party"),
    ("COND", r"frustration", "ma.condition.frustration"),
    ("COND", r"representations|accuracy|bring.?down", "ma.condition.reps_bringdown"),
    ("COND", r"performance|covenants|compliance with (agreement|obligations)", "ma.condition.covenant_performance"),
    ("COND", r"material adverse|no (adverse )?change", "ma.condition.no_mae"),
    ("COND", r"stockholder|shareholder|vote", "ma.condition.stockholder_approval"),
    ("COND", r"injunction|restraint|illegality|legal (prohibition|impediment)|orders?$|no (law|prohibition|order)", "ma.condition.no_legal_restraint"),
    ("COND", r"hsr|antitrust|regulatory|governmental|approvals|waiting period|cfius", "ma.condition.regulatory_approval"),
    ("COND", r"registration|form s-4|listing|nyse|nasdaq", "ma.condition.registration_listing"),
    ("COND", r"financing", "ma.condition.financing"),
    ("COND", r"dissent|appraisal", "ma.condition.dissenting_shares"),
    ("COND", r"consents", "ma.condition.third_party_consents"),
    ("COND", r"litigation|proceedings|actions|suits", "ma.condition.no_litigation"),
    ("COND", r"tax opinion|opinion.*tax", "tax.opinion.closing_condition"),
    ("COND", r"minimum cash|available cash|trust", "ma.spac.minimum_cash"),
    ("COND", r"certificate|deliver|documents|opinion|resignation|agreements|employ|closing|waiver|escrow|payoff|release|other|satisfaction|firpta|good standing", "ma.condition.other"),
    # ---- termination (context)
    ("TERM", r"effect of|effects of|consequences|liabilit|obligations (upon|on|after)|survival", "ma.termination.effect"),
    ("TERM", r"(termination|break.?up|reverse|parent|company)? ?fees?( and expenses)?$|termination (fee|payment)|fees and expenses|payments", "ma.nav.termination.fees"),
    ("TERM", r"expense", "ma.termination.expense_reimbursement"),
    ("TERM", r"mutual (consent|agreement)", "ma.termination.right.mutual_consent"),
    ("TERM", r"notice|procedure", "ma.termination.notice"),
    ("TERM", r"^termination( of (this |the )?agreement)?$|termination (rights|events|by|prior)|right(s)? (to|of) terminat|grounds|abandonment|^terminat", "ma.nav.termination.rights"),
    ("TERM", r"exclusive remed|sole remed|limitation(s)? on (liability|remedies)", "ma.termination.fee.exclusive_remedy"),
    # ---- indemnification (context)
    ("INDEM", r"survival|non-?survival|expiration of|time limit|limitation period", "ma.indemn.survival"),
    ("INDEM", r"third.?party claims|defense|control of|assumption of defense", "ma.indemn.third_party_claims"),
    ("INDEM", r"procedure|notice of claims|claims|notification|method of asserting|direct claims|resolution of", "ma.indemn.claims_procedure"),
    ("INDEM", r"exclusive|sole remed|remedies$|exclusivity", "ma.indemn.exclusive_remedy"),
    ("INDEM", r"limit|cap\b|basket|threshold|deductible|maximum|minimum|de minimis|certain limitations", "ma.nav.indemnification.limits"),
    ("INDEM", r"escrow|holdback|hold-back|retention amount", "ma.indemn.escrow"),
    ("INDEM", r"set-?off|offset|recoupment", "ma.indemn.setoff"),
    ("INDEM", r"mitigat", "ma.indemn.mitigation"),
    ("INDEM", r"insurance (proceeds|recover|and)|net (of|losses)|tax benefit|recover|subrogation|no double|duplication|calculation of (losses|damages)|determination of (loss|damages)", "ma.indemn.recovery_offsets"),
    ("INDEM", r"tax treatment|treatment of (indemni|payments)|adjustment(s)? to (the )?(purchase price|merger consideration)|characterization|purchase price adjustment", "tax.adjustment.price_adjustments"),
    ("INDEM", r"investigation|knowledge|reliance|sandbag|waiver of conditions", "ma.indemn.sandbagging"),
    ("INDEM", r"materiality", "ma.indemn.materiality_scrape"),
    ("INDEM", r"r&w|rwi|r & w|representation(s)? and warrant(y|ies) (insurance|policy)|buyer.side policy|insurance policy", "ma.indemn.rwi"),
    ("INDEM", r"representative|stockholders agent|sellers agent", "ma.seller_representative"),
    ("INDEM", r"source|order of|manner of payment|payment|satisfaction of|recourse", "ma.indemn.order_of_recovery"),
    ("INDEM", r"special|specific|environmental|certain (matters|indemn)", "ma.indemn.specific"),
    ("INDEM", r"tax", "tax.nav.indemnity"),
    ("INDEM", r"indemni|obligation|agreement to|general|by (the )?(seller|buyer|purchaser|parent|company|stockholders|shareholders|members)", "ma.indemn.obligation"),
    # ---- tax article (context)
    ("TAX", r"transfer tax|conveyance|sales (and use )?tax", "tax.transfer.allocation"),
    ("TAX", r"straddle|allocation of (taxes|certain taxes|tax)|apportionment|proration", "tax.nav.indemnity.straddle"),
    ("TAX", r"refund|credits", "tax.nav.compliance.refunds"),
    ("TAX", r"contest|audits?|proceedings|controvers|claims|disputes", "tax.nav.compliance.contests"),
    ("TAX", r"returns|filing", "tax.nav.compliance.returns"),
    ("TAX", r"cooperat|assistance|records|information", "tax.nav.compliance.post_closing_cooperation"),
    ("TAX", r"indemni|liability for taxes|responsibility for taxes", "tax.nav.indemnity"),
    ("TAX", r"338|336|election", "tax.nav.deal.elections"),
    ("TAX", r"allocation", "tax.nav.deal.allocation"),
    ("TAX", r"sharing|allocation agreement", "tax.cooperation.tax_sharing_termination"),
    ("TAX", r"treatment|reorgani[sz]ation|intended|characterization|tax-free|structure", "tax.nav.deal.treatment"),
    ("TAX", r"withhold|firpta|certificate", "tax.nav.transfer_withholding.withholding"),
    ("TAX", r"post-closing (actions|transactions)|amended returns|certain actions", "tax.cooperation.post_closing_actions"),
    ("TAX", r"survival|coordination|conflict|exclusiv|overlap|limitations", "tax.nav.indemnity.mechanics"),
    ("TAX", r"definitions|certain defined", "tax.nav.definitions"),
    ("TAX", r"tax", "tax.nav.compliance"),
    # ---- definitions (context)
    ("DEF", r"material adverse", "ma.mae.definition"),
    ("DEF", r"knowledge", "ma.def.knowledge"),
    ("DEF", r"disclosure (schedule|letter)|schedules", "ma.interpretation.disclosure_schedules"),
    ("DEF", r"interpret|construction|other definitional|headings|usage|references|rules of|certain (interpretations|matters of construction)|terms generally|accounting terms|gender|currency", "ma.interpretation.rules"),
    ("DEF", r"defin|certain terms|terms$|cross.?reference|index of|glossary|^[a-z].{0,60} (means|shall mean)\b|^person$|^affiliate$", "ma.definitions.general"),
    # ---- global rules: structure and closing
    (None, r"^the merger(s)?$|^merger$|^the (first|second|initial|subsequent|company|bank|parent|holdco|reincorporation|upstream) merger|mergers? and|^the (share )?exchange$|^the (transaction|combination|arrangement|amalgamation|scheme)s?$|bank merger|second merger|subsidiary merger|plan of merger|constituent|^the (business )?combination", "ma.structure.merger"),
    (None, r"purchase and sale|sale and purchase|sale of (the )?(shares|stock|interests|units|securities|membership|purchased|acquired|company)|purchase of (the )?(shares|stock|interests|units|securities|membership|company)|transfer of (the )?(shares|interests|units|stock)|agreement to (purchase|sell)|^(the )?(sale|purchase|acquisition|contribution)$|contribution and exchange|^sale and transfer", "ma.structure.purchase_and_sale"),
    (None, r"effective (time|date)|certificate(s)? of merger|articles of merger|filing of|consummation of the merger", "ma.structure.effective_time"),
    (None, r"effects? of (the )?(merger|mergers|transaction|closing)s?$|effects? of (the )?mergers? on (the )?(constituent|surviving)|general effects", "ma.structure.effects_of_merger"),
    (None, r"(certificate|articles) of (incorporation|formation|organization)|by-?laws|charter|organizational documents|governing documents|operating agreement|memorandum and articles|limited liability company agreement", "ma.structure.governing_documents"),
    (None, r"directors and officers of|directors;? (and )?officers$|^(directors|officers|managers|board of directors)( and (officers|managers|management))?$|officers and directors of|(directors|officers|managers) of (the )?(surviving|company|merger sub)|board of directors of (the )?surviving|management of the surviving|^directors of", "ma.structure.directors_officers"),
    (None, r"alternative (structure|transaction structure)|change (of|in) (method|structure)|restructur.*(transaction|merger)|structure of (the )?(merger|transaction)|modification of structure", "ma.structure.alternative_structure"),
    (None, r"pre-?closing (reorgani[sz]ation|restructuring|transactions)|^reorgani[sz]ation$|^restructuring$|preliminary transactions|pre-?merger transactions|^conversion$|internal reorgani", "tax.restructuring.pre_closing"),
    (None, r"^(purchased|acquired|transferred|conveyed|contributed|included) assets|assets to be (purchased|acquired|sold|transferred|conveyed)|(purchase|sale|transfer|conveyance|acquisition) (and sale )?of (the )?(purchased |acquired |transferred )?assets|^assets$|sale of assets", "ma.asset.purchased_assets"),
    (None, r"excluded assets|retained assets|assets not (being )?(transferred|acquired|sold)", "ma.asset.excluded_assets"),
    (None, r"excluded liabilit|retained liabilit|liabilities not assumed|no (other )?assumption|non-?assumed", "ma.asset.excluded_liabilities"),
    (None, r"assum(ed|ption of) (the )?(certain )?(liabilit|obligations)|^liabilities$|assumption$|liabilities (to be )?assumed", "ma.asset.assumed_liabilities"),
    (None, r"non-?assignab|non-?transferab|assignment of (contracts|certain)|consents? to assign|nonassignable|unassigned|restricted (assets|contracts)|delayed (assets|transfer)|deferred (transfer|closing|assets)|shared contracts", "ma.asset.nonassignable_assets"),
    (None, r"bulk (sales|transfer)", "ma.asset.bulk_sales"),
    (None, r"risk of loss|casualty|condemnation|damage or destruction|damage to", "ma.asset.risk_of_loss"),
    (None, r"^the offer$|^offer$|tender offer|exchange offer|terms of the offer|offer documents|conduct of the offer|commencement of the offer|expiration of the offer|the offer and", "ma.tender.offer"),
    (None, r"conditions (to|of) the offer|offer conditions|minimum condition", "ma.tender.offer_conditions"),
    (None, r"company actions?$|actions by the company|schedule 14d-9|company action", "ma.tender.company_actions"),
    (None, r"top-?up", "ma.tender.top_up_option"),
    (None, r"without (a )?(meeting|vote)|251\(h\)|short.?form|merger without", "ma.tender.back_end_merger"),
    (None, r"14d-10", "ma.tender.rule_14d10"),
    (None, r"^directors$|board (representation|designees|of directors;)|14\(f\)|14f-1|composition of (the )?(company )?board|directors following", "ma.tender.board_designees"),
    (None, r"deliver(ies|ables|y)|closing (documents|actions|obligations|transactions|certificates?)|items to be delivered|documents to be delivered|actions (at|to be taken at) (the )?closing|transactions to be effected at|closing agreements|at the closing|obligations at closing", "ma.closing.deliverables"),
    (None, r"closing payments?|payments? at (the )?closing|flow of funds|payment of (the )?(purchase price|closing|consideration|indebtedness|transaction expenses)|closing date payments|payments on the closing|use of proceeds|^payments?$|closing (cash )?consideration payment|payment and delivery", "ma.closing.payments"),
    (None, r"^(the )?closing(s)?( date)?$|time and place|closing;|^closing (date|time)|date of closing|place of closing|closing and effective|the closing|closing of the (merger|transaction)|^closing$|consummation of the (transaction|closing)|^closing ", "ma.closing.time_and_place"),
    (None, r"trust (account|fund) waiver|waiver of (claims against|access to) trust|claims against (the )?trust|no claim against (the )?trust|waiver.*trust|trust.*waiver", "ma.spac.trust_waiver"),
    (None, r"trust (account|fund|agreement)", "ma.spac.trust_account"),
    (None, r"pipe|subscription agreements?|forward purchase|private placement", "ma.spac.pipe"),
    (None, r"minimum cash|available closing cash", "ma.spac.minimum_cash"),
    (None, r"redemption", "ma.spac.redemption"),
    (None, r"sponsor", "ma.spac.sponsor_support"),
    (None, r"extension (of|proposal)|business combination deadline", "ma.spac.extension"),
    # ---- consideration
    (None, r"exchange ratio", "ma.consideration.exchange_ratio"),
    (None, r"election|proration", "ma.consideration.election_proration"),
    (None, r"collar|walk.?away", "ma.consideration.exchange_ratio"),
    (None, r"earn.?outs?|contingent (consideration|payment|purchase price)|milestone|additional consideration|deferred (consideration|payment|purchase price)|contingent merger", "ma.consideration.earnout"),
    (None, r"contingent value|cvr", "ma.consideration.cvr"),
    (None, r"deposit|earnest money", "ma.consideration.deposit"),
    (None, r"dilution|adjustments? (to|of) (the )?(exchange ratio|merger consideration|conversion)|certain adjustments|equitable adjustments?|^adjustments?$|changes in (company )?(stock|capitali[sz]ation)|stock splits|recapitali[sz]ation|reclassification", "ma.consideration.anti_dilution"),
    (None, r"rollover|roll-over|reinvestment", "ma.consideration.rollover"),
    (None, r"lock-?up|registration rights|legends?$|restrictive legend|restricted (stock|shares|securities)|resale|rule 144|seller notes?|promissory note|transfer restrictions|restrictions on transfer|securities (laws|act) (matters|compliance|exemption)|private placement|registration of (shares|parent|buyer)|form s-[38]|shelf registration|stock consideration", "ma.consideration.buyer_securities"),
    (None, r"(stock |share |equity |incentive )?(options?|rsus?|psus?|restricted stock|equity( and equity-based)? awards?|stock (plans?|awards|appreciation|incentive|based)|equity (plans?|incentive|compensation|interests)|performance (shares|units|awards)|phantom|sars?|profits interests|incentive units|deferred (stock|units)|equity-based|share (awards|plans))", "ma.equity.awards"),
    (None, r"espp|stock purchase plan", "ma.equity.espp"),
    (None, r"warrants|convertible|preferred (stock|shares)|notes$|debentures|company securities|other securities|treatment of (company )?(securities|notes)", "ma.equity.warrants_convertibles"),
    (None, r"dissent|appraisal", "ma.exchange.dissenting_shares"),
    (None, r"fractional", "ma.exchange.fractional_shares"),
    (None, r"allocation (schedule|statement|certificate)|payment (spreadsheet|schedule)|consideration (spreadsheet|schedule)|distribution (schedule|waterfall)|closing (payment )?certificate|^spreadsheet|payout (spreadsheet|schedule)|closing consideration schedule|pro rata", "ma.exchange.payment_schedule"),
    (None, r"exchange (of|agent|fund|procedures)|paying agent|payment (of|for) (the )?(merger consideration|shares|securities|company)|surrender|letters? of transmittal|lost|stolen|transfer books|stock transfer|no further (ownership|rights|transfers)|unclaimed|exchange and payment|payment procedures|delivery of (the )?(merger )?consideration|termination of (the )?(exchange|payment) fund|investment of (the )?(exchange|payment) fund|book-?entry|certificates|distributions with respect to unexchanged|abandoned property|escheat|payment fund|further rights|closing of (the )?(company.?s? )?(stock|share|transfer)|payment mechanics|uncertificated", "ma.exchange.procedures"),
    (None, r"withholding|deductions from", "tax.withholding.provision"),
    (None, r"conversion of|effect on (capital|shares|securities|stock|company|outstanding|membership)|effect of (the )?mergers? on|merger consideration|consideration$|consideration for|^consideration|treatment of (shares|capital stock|common stock|company (common )?stock|units)|cancellation of|capital stock of|per share|(company|merger sub|parent|target) (common )?(stock|shares|capital stock|units)$|shares of|status of (shares|capital)|stock of merger sub|aggregate consideration|closing consideration|total consideration", "ma.consideration.merger_consideration"),
    (None, r"working capital|closing (date )?(balance sheet|statement)|estimated closing|post-?closing (adjustment|statement|purchase price|determination|true)|purchase price adjustment|adjustment (to|of) (the )?(purchase price|consideration|cash|closing|base)|final (purchase price|closing|adjustment|determination)|preliminary (closing|purchase price)|closing adjustment|price adjustment|determination of (final|closing|purchase price|net working|adjustment)|true-?up|net (asset|worth|cash|debt) adjustment|estimated (purchase price|adjustment|statement)|adjustment (amount|procedure|payment)|calculation of (purchase price|closing|final|adjustment)", "ma.price_adjustment.mechanism"),
    (None, r"independent account|accounting (firm|arbitrator|referee)|neutral (account|arbit)|resolution of (disputes|objections)|dispute(d)? (items|resolution procedure)|disputes regarding", "ma.price_adjustment.dispute_resolution"),
    (None, r"adjustment (escrow|holdback)", "ma.price_adjustment.escrow"),
    (None, r"purchase price$|^purchase price|^price$|amount of (purchase price|consideration)|base (purchase price|consideration)|calculation of (the )?(merger )?consideration|payment of purchase price|consideration and payment|purchase consideration|^consideration|unit purchase price", "ma.consideration.purchase_price"),
    (None, r"allocation of (the )?(purchase price|consideration|merger consideration)|purchase price allocation|price allocation|^allocation$|tax allocation|asset allocation|allocation for tax", "tax.nav.deal.allocation"),
    # ---- conduct and access
    (None, r"no control|control of (the )?(other|company|operations|business)", "ma.conduct.no_control"),
    (None, r"conduct of (the )?(company|business|seller|target|parent|purchaser|buyer|acquiror|operations)|operation(s)? of (the )?(company|business|seller|target)|interim (operations|conduct|covenants|period)|ordinary course|covenants of (the )?(company|seller|parent|buyer)|conduct (prior|pending|until)|negative covenants|affirmative covenants|forbearances|pre-?closing (covenants|conduct|operations|actions)|restrictions on (the )?(company|business|conduct|seller)|business (covenants|conduct)|preservation of business|maintenance of business|carry on|restricted actions|certain covenants of|required actions|prohibited actions|company (covenants|forbearances|conduct)|parent (forbearances|conduct|covenants)|operating covenants|^covenants$|actions pending", "ma.conduct.ordinary_course"),
    (None, r"schedule (updates?|supplements?)|(update|supplement|amendment)s? (of|to) (the )?(disclosure )?schedules|supplements? to (disclosure|schedules)|supplemental disclosure|updating|disclosure supplements|updates to", "ma.access.schedule_updates"),
    (None, r"notif|notices? of (certain|developments|events|breach|changes|claims|actions)|advice of changes|notice of developments|certain notices|notice to (parent|buyer|purchaser|other)|advise|current information|reporting", "ma.access.notification"),
    (None, r"confidential|non-?disclosure|nondisclosure", "ma.access.confidentiality"),
    (None, r"public|press|publicity|announcement|communications|media", "ma.access.public_announcements"),
    (None, r"access|inspection|investigation|information$|due diligence|examination|books and records|furnishing|availability of|review of|monthly (financial|reports)|financial (statements|information|reports)|delivery of (financial|reports)|interim financial", "ma.access.information"),
    # ---- deal protection and approval
    (None, r"go-?shop", "ma.noshop.go_shop"),
    (None, r"standstill", "ma.noshop.standstill_waiver"),
    (None, r"change (of|in) (the )?(company )?(board )?recommendation|adverse recommendation|recommendation change|change of board|fiduciary (out|duties|obligations)|superior (proposal|offer)|board recommendation|recommendation of (the )?(company|board)|company (board )?recommendation|company recommendation", "ma.nav.deal_protection.recommendation"),
    (None, r"no.?shop|no (solicitation|negotiation|other negotiations|other bids|inconsistent|alternative|other offers|other discussions)|non-?solicitation of (alternative|acquisition|competing|other|transactions|proposals|bids|offers)|acquisition proposals?|alternative (proposals?|transactions?)|competing (proposals?|transactions?|offers?)|other (offers|proposals|bids|transactions)|exclusiv|takeover proposals?|solicitation$|^no solicitation|^non-?solicitation$|exclusive dealing|third party proposals|unsolicited|other potential", "ma.noshop.non_solicitation"),
    (None, r"written consent|consent of (stockholders|shareholders|members)|information statement|stockholder consent|shareholder consent|requisite (stockholder )?consent|consent solicitation|stockholder notice", "ma.approval.written_consent"),
    (None, r"proxy|registration statement|form s-4|form f-4|joint proxy|sec filings|preparation of|prospectus|circular|schedule 13e|scheme document|disclosure documents", "ma.approval.proxy_statement"),
    (None, r"(stockholder|shareholder|member|unitholder|special)s?.? (meeting|approval|vote|action)|meetings? of (the )?(company )?(stockholder|shareholder)|company meeting|approval (of|by) (the )?(company )?(stockholder|shareholder|member)|parent (stockholder|shareholder) (meeting|approval)|stockholders meeting|shareholders meeting|^meeting|vote of|general meeting|company (stockholder|shareholder)", "ma.approval.stockholder_meeting"),
    (None, r"voting (agreement|and support|undertaking)|support (agreement|of transaction)|stockholder (agreement|support)|tender (and support|agreement)|shareholder (agreement|support)|lock-up agreement|agreement to vote|voting of shares|irrevocable (proxy|undertaking)|^voting$|voting (matters|commitment)", "ma.approval.voting_agreements"),
    (None, r"takeover|anti-?takeover|state (takeover|anti)|business combination (statute|law)|section 203|control share|moratorium|fair price", "ma.takeover.statutes"),
    (None, r"rights (agreement|plan)|poison pill|rights amendment", "ma.takeover.rights_plan"),
    # ---- regulatory
    (None, r"filing fees|hsr fees", "ma.regulatory.filing_fee_allocation"),
    (None, r"cfius|foreign investment|exon.florio|national security|dpa\b|facility clearance|foci\b", "ma.regulatory.cfius"),
    (None, r"hsr|antitrust|competition (law|filing)|hart-scott|merger control|premerger", "ma.regulatory.antitrust_filings"),
    (None, r"divestiture|regulatory remed", "ma.regulatory.remedies"),
    (None, r"regulatory (matters|approvals|filings|applications|and other|consents|authori[sz]ations|compliance|efforts|undertakings|cooperation)|governmental (approvals|filings|consents|authori[sz]ations|reviews)|(government|governmental|required) (approvals|filings)|applications|filings; (other|reasonable)|certain filings|regulatory$|approvals of governmental|legal conditions to|fcc|insurance regulatory|bank regulatory|state regulatory|gaming|necessary (approvals|filings)|other (regulatory|approvals)|filings$|approvals$|approvals and consents|requisite regulatory|permits$|transfer of permits|licenses and permits", "ma.regulatory.industry_approvals"),
    (None, r"(third.?party |required |contract |necessary |certain |obtaining )?consents?( and (waivers|approvals|notices))?$|consents of third parties|third party (approvals|notices|consents)|consents; |notices and consents|landlord|estoppel|^consents? ", "ma.efforts.third_party_consents"),
    (None, r"efforts|cooperation$|commercially reasonable|^cooperation|appropriate action|consummation|satisfaction of (closing )?conditions|fulfillment of conditions|agreement to cooperate|actions to consummate|reasonable (best|commercial)|conditions to closing$|mutual (cooperation|assistance)|closing conditions$|taking of necessary action|necessary action|further action; (reasonable|efforts)|implementing agreement|agreement to (take|use)|additional agreements$|^general$", "ma.efforts.general"),
    # ---- financing
    (None, r"financing sources?|lender|debt financing source|financing parties|financing provisions|certain financing|waiver of claims against financing|debt financing matters$", "ma.financing.source_protections"),
    (None, r"financing (cooperation|assistance)|cooperation with (the )?(debt )?financing|assistance with financing|company cooperation|cooperation (in connection|regarding) with (the )?financing|financing; cooperation", "ma.financing.cooperation"),
    (None, r"payoff|pay-off|repayment of|termination of (existing|credit|indebtedness)|existing (indebtedness|credit|debt|notes)|treatment of (existing |certain |company )?(indebtedness|debt|notes|credit)|release of (liens|guarant|encumbrances)|lien releases?|indebtedness|credit (facility|agreement)|senior notes|debt tender|notes (tender|redemption)|consent solicitation|indentures?|redemption of notes|^debt$|company (notes|indebtedness|debt)|discharge of|guarantees?$|release of guarant|replacement of guarant|letters of credit|credit support|surety|bonds", "ma.financing.payoff_letters"),
    (None, r"equity (commitment|financing)|sponsor commitment", "ma.financing.equity_commitment"),
    (None, r"guarant(ee|y|or)|parent undertaking|performance guarant", "ma.financing.limited_guarantee"),
    (None, r"debt (financing|commitment)|commitment letters?", "ma.financing.debt_commitment"),
    (None, r"financing|funding|^funds$|available funds", "ma.financing.buyer_covenant"),
    # ---- employees
    (None, r"280g|parachute|excess payments", "ma.employee.section_280g"),
    (None, r"warn\b|plant closing", "ma.employee.warn"),
    (None, r"401\(?k|termination of (certain )?(company |employee )?(benefit |pension |retirement |savings )?plans?|plan terminations?|retirement plans?|savings plan|pension plans?|qualified plans?", "ma.employee.plan_termination"),
    (None, r"severance|retention|bonus|incentive (plans?|compensation|payments)|change (in|of) control (payments|agreements)|key employee|employment (agreements?|arrangements|contracts)|non-?competition agreements|new employment|management (agreements|arrangements)|offer letters", "ma.employee.retention_severance"),
    (None, r"(transferred|offered|business|hired|continuing|affected) employees|offers? of employment|employment of|transfer of employees|employees and offers|employee transfers?|hiring of|employment offers", "ma.employee.transfer"),
    (None, r"employee|benefit|personnel|labor|compensation|workers|cobra|vacation|union|collective bargaining|erisa|employment|human resources|welfare plans|service credit|health (plans|coverage)|plans$", "ma.nav.employees.continuing"),
    # ---- other covenants
    (None, r"(d&o|directors.? and officers.?|director and officer|officers.? and directors.?|d & o) (insurance|liability insurance|tail)|tail (insurance|policy)|run-?off", "ma.dando.tail_insurance"),
    (None, r"indemnification(;|,| and| of)? ?(exculpation|insurance|directors|officers|d&o|of (company )?(directors|officers|d&o))|(d&o|directors.? and officers.?|director and officer|officers.? and directors.?|officer and director|d & o)|indemnification and insurance|indemnification; (directors|insurance|exculpation)|exculpation|insurance and indemnification|indemnification of (company |certain |covered |former )?(directors|officers|persons|parties|indemnitees)|directors and officers$|continuing indemnification|indemnification continuation|indemnification$|indemnification (by|of) (parent|surviving|the surviving)|liability insurance|indemnity; insurance|indemnification obligations", "ma.dando.indemnification"),
    (None, r"section 16|16\(b\)|rule 16b-3|16b-3", "ma.governance.section_16"),
    (None, r"resignation", "ma.governance.resignations"),
    (None, r"governance|headquarters|corporate (name|offices|governance)|name (of|and) (the )?(combined|surviving|parent)|board (composition|of directors of parent|seats|matters|appointment|membership)|post-?closing (board|directors|governance|management)|parent board|directors of parent|appointment of (directors|director)|composition of (the )?(parent )?board|chief executive|executive (officers|chairman)|management (of|succession)|advisory board|name change|parent (directors|governance)|board representation|community (commitments|matters)|charitable|combined company", "ma.governance.post_closing"),
    (None, r"listing|delist|de-list|deregistration|nyse|nasdaq|stock exchange|exchange act (de)?registration|quotation|nyse american|toronto stock", "ma.listing.stock_exchange"),
    (None, r"dividend|distributions$", "ma.dividend_coordination"),
    (None, r"(stockholder|shareholder|transaction|securityholder|securities|certain|deal|merger|unitholder) litigation|litigation (support|cooperation|matters)$|defense of litigation|stockholder suits|transaction-related litigation|litigation$|shareholder (suits|claims)", "ma.transaction_litigation"),
    (None, r"non-?compet|noncompet|covenant not to compete|restrictive covenants?|competition$|competing business|restricted (business|activities)|non-?competition; non-?solicitation", "ma.restrictive.non_compete"),
    (None, r"non-?solicit|no (hire|hiring|solicitation of employees|raid)|non-?hire|non-?disparage|solicitation of employees|employee non-?solicit|no solicitation or hiring|nonsolicitation", "ma.restrictive.non_solicit"),
    (None, r"^releases?$|general release|release of (claims|company|seller|parties|certain)|mutual release|waiver and release|release by|releases? and|seller release|stockholder release|waiver of claims|^release ", "ma.release"),
    (None, r"further (assurance|action|assistance|agreement|documents|instruments)|additional (documents|actions|instruments|assurances)|subsequent actions|other actions|post-?closing (cooperation|covenants|obligations|matters|actions|assistance)|further$|additional agreements|necessary further|post closing", "ma.further_assurances"),
    (None, r"(preservation|retention|maintenance) of (books|records)|books and records|access to (books|records)|records$|record retention|post-?closing access|litigation (support|cooperation|assistance)|cooperation in litigation|access after|retention of (books|records)|preservation of", "ma.records_access"),
    (None, r"wrong pocket|misdirected|misallocated|receipt of (payments|funds)|collection of (accounts|receivables)|payments received|accounts receivable|mail|refunds and remittances|remittance|turnover|post-?closing (receipts|payments|collections)", "ma.wrong_pockets"),
    (None, r"use of (the )?(seller |certain |retained |corporate |company |business |parent )?(name|marks|trademark|trade name|intellectual)|names? and marks|corporate name|seller (names|marks)|retained (names|marks)|trade names?|name change|change of name|trademarks?$|licensed marks|transitional (trademark|use)", "ma.use_of_name"),
    (None, r"transition(al)? (services|arrangements|matters|assistance|plan)|migration|^transition$|tsa\b", "ma.transition_services"),
    (None, r"intercompany|inter-company|affiliate (agreements|arrangements|contracts|transactions|matters)|related party (agreements|contracts|transactions)|termination of (certain |affiliate |related |intercompany |stockholder |shareholder |investor |existing )?(agreements|contracts|arrangements|rights)|stockholders? agreement|shareholders? agreement|investor rights|termination of agreements|certain (agreements|arrangements|contracts)$", "ma.intercompany_termination"),
    (None, r"r&w|rwi|r & w|representations? and warrant(y|ies) (insurance|policy)|buyer.side (policy|insurance)|insurance policy$|rep and warranty", "ma.indemn.rwi"),
    (None, r"representative|stockholders.? agent|sellers.? agent|securityholders.? agent|shareholders.? agent|holder (agent|rep)|equityholder rep|agent for|appointment of (seller|stockholder|shareholder|agent)", "ma.seller_representative"),
    (None, r"escrow|holdback|hold-back|retention (amount|fund|holdback)|expense fund", "ma.indemn.escrow"),
    (None, r"set-?off|offset|recoupment", "ma.indemn.setoff"),
    (None, r"exclusive remed|sole (and exclusive )?remed|exclusivity of remed|remedies exclusive|exclusive (nature|means)|sole remedy", "ma.indemn.exclusive_remedy"),
    (None, r"non-?survival|no survival|nonsurvival|survival|expiration of representations|termination of representations", "ma.indemn.survival"),
    (None, r"third.?party claims|defense of (claims|third)|claims (procedure|process)|indemnification (procedure|claims|process)|procedures? for (indemnification|claims)|notice of claims|^claims$|direct claims|conduct of claims|handling of claims|matters involving third parties|procedures relating to", "ma.nav.indemnification.procedure"),
    (None, r"limitations? (on|of|to) (indemni|liability|recovery|claims|damages|remedies|amount)|certain limitations|^limitations?$|caps?$|basket|threshold|deductible|maximum (liability|indemn)|indemnification (limits|limitations|cap)|liability (cap|limits)|limits on|de minimis|other limitations|limitation of indemnification|limitation(s)? and", "ma.nav.indemnification.limits"),
    (None, r"mitigat", "ma.indemn.mitigation"),
    (None, r"materiality (scrape|qualifiers)|disregard of materiality", "ma.indemn.materiality_scrape"),
    (None, r"(effect|no effect) of (investigation|knowledge|waiver)|reliance|sandbag|knowledge of breach|investigation$|no waiver by investigation", "ma.indemn.sandbagging"),
    (None, r"(tax )?treatment of (indemni|payments|indemnity)|indemni(ty|fication) payments|characterization of (indemn|payments)|adjustment to (purchase price|merger consideration)$|tax (treatment|characterization) of", "tax.adjustment.price_adjustments"),
    (None, r"indemnification (by|of|obligations of)|indemnity (by|of)|indemnification$|indemnities$|indemnify|obligations? to indemnify|general indemn|agreement to indemnify|indemnification (generally|rights|agreement)|^indemnity$|(seller|buyer|purchaser|parent|company|stockholder|shareholder).?s? indemn|indemnification;", "ma.indemn.obligation"),
    (None, r"insurance (proceeds|recover|and other)|net (losses|recovery)|subrogation|no double recovery|duplication|calculation of (losses|damages)|determination of (loss|damages)|tax benefits?$|recoveries|reduction (of|in) (losses|amount)|damages$|losses$|payment of (claims|indemn)|manner of payment|satisfaction of (claims|indemn)|order of recovery|source of recovery|recourse|payments$", "ma.indemn.recovery_offsets"),
    # ---- tax (global)
    (None, r"transfer tax|conveyance tax|sales (and|,) use|documentary|stamp (tax|dut)|real (estate|property) transfer|sales tax|transfer and (gains|sales)|transaction taxes|recording fees", "tax.transfer.allocation"),
    (None, r"338|336\(e\)|section 754", "tax.nav.deal.elections"),
    (None, r"straddle|proration of taxes|apportionment of taxes|allocation of (certain )?taxes|property taxes|prorations?$|apportion", "tax.nav.indemnity.straddle"),
    (None, r"tax refund|refunds?$|refunds and (tax|credits)", "tax.nav.compliance.refunds"),
    (None, r"tax (contests?|audits?|proceedings|claims|controvers|disputes)|contests?$|audits$", "tax.nav.compliance.contests"),
    (None, r"tax returns?|filing of (tax )?returns|preparation of tax|returns$", "tax.nav.compliance.returns"),
    (None, r"tax (sharing|allocation) agreements?", "tax.cooperation.tax_sharing_termination"),
    (None, r"tax indemn|indemnification for taxes|liability for taxes|tax (liability|liabilities)", "tax.nav.indemnity"),
    (None, r"tax cooperation|cooperation on tax|cooperation (and|in|with respect to) .*tax|tax assistance|assistance and cooperation", "tax.nav.compliance.post_closing_cooperation"),
    (None, r"firpta|w-9|non-?foreign|tax (certificates?|forms)|1445", "tax.withholding.form_delivery"),
    (None, r"tax opinions?|tax representation letters|opinions? of counsel|tax certificates", "tax.nav.deal.opinions"),
    (None, r"tax (treatment|consequences|characterization|free|structure|status|reporting|classification|qualification)|tax-free|reorgani[sz]ation|plan of reorgani|intended tax|section 368|qualification as a|tax-deferred|treatment as|u\.s\. tax treatment|income tax treatment|federal (income )?tax|351", "tax.nav.deal.treatment"),
    (None, r"reit|real estate investment trust", "tax.passthrough.reit"),
    (None, r"net operating loss|nol|382|tax attributes|tax benefit preservation", "tax.nav.attributes"),
    (None, r"^tax|taxes|taxation|tax (matters|covenants|provisions)", "tax.nav.compliance"),
    # ---- conditions (global)
    (None, r"frustration", "ma.condition.frustration"),
    (None, r"conditions? (precedent )?(to|of) (the )?(obligations? of )?(each|all|both|the parties|every)|mutual conditions|conditions to (the )?(merger|closing|each|obligations of the parties)$|joint conditions|conditions (applicable )?to all|general conditions|conditions to (the )?respective", "ma.nav.conditions.mutual"),
    (None, r"conditions?|officer.?s certificate|bring.?down|closing certificate", "ma.nav.conditions.party"),
    # ---- termination (global)
    (None, r"(effect|effects|consequences) of (termination|abandonment)|termination; effect|effect of the termination|liabilities in event of termination|obligations (upon|on) termination|survival (after|upon) termination", "ma.termination.effect"),
    (None, r"expense reimbursement|reimbursement of expenses|expenses (following|upon|on|in the event of) termination", "ma.termination.expense_reimbursement"),
    (None, r"reverse (termination|break)", "ma.termination.fee.reverse"),
    (None, r"(termination|break.?up|breakup|topping) (fees?|payment|amount)|fees? (and expenses )?(upon|following|on) termination|company termination|parent termination|termination; fees|expenses; termination fees?|fees and expenses; termination|termination expenses|liquidated damages", "ma.nav.termination.fees"),
    (None, r"notice of termination|procedure (for|upon|and effect of) termination|termination procedure|method of termination|manner of termination", "ma.termination.notice"),
    (None, r"outside date|end date|drop dead", "ma.termination.outside_date"),
    (None, r"terminat|abandonment|rescission", "ma.nav.termination.rights"),
    # ---- remedies
    (None, r"governing law|applicable law|choice of law|controlling law|law governing|governing$|^law$", "ma.governing_law"),
    (None, r"jurisdiction|venue|forum|consent to service|service of process|submission to|courts|agent for service", "ma.forum_selection"),
    (None, r"jury", "ma.jury_waiver"),
    (None, r"arbitrat|mediat|dispute resolution|resolution of disputes|disputes$|alternative dispute|^dispute", "ma.arbitration"),
    (None, r"specific (performance|enforcement)|injunctive|equitable (relief|remedies)|enforcement( of (this )?agreement)?$|injunction|remedies; specific|irreparable", "ma.specific_performance"),
    (None, r"consequential|punitive|limitation (of|on) (liability|damages)|no (consequential|special|punitive)|waiver of (certain )?damages|damages waiver|exclusion of|special damages|certain damages|limited liability|limitations on liability|no liability|limitation of liability", "ma.damages_waiver"),
    (None, r"non-?recourse|no recourse|no personal liability|recourse|no liability of|personal liability|limited recourse", "ma.non_recourse"),
    (None, r"attorney|legal fees|prevailing party|costs of enforcement|enforcement costs|collection (costs|expenses)", "ma.attorneys_fees"),
    (None, r"remedies|cumulative|rights and remedies|other remedies|no election", "ma.remedies_cumulative"),
    # ---- definitions and interpretation
    (None, r"material adverse", "ma.mae.definition"),
    (None, r"knowledge", "ma.def.knowledge"),
    (None, r"disclosure (schedules?|letters?|memorandum)|schedules$|exhibits and schedules|schedules and exhibits|exhibits$|incorporation of (schedules|exhibits|annexes)|annexes|schedules; |company disclosure|seller disclosure|parent disclosure|^disclosure$|appendices|attachments", "ma.interpretation.disclosure_schedules"),
    (None, r"interpret|construction|headings|captions|usage|rules of|references|gender|number$|other definitional|terms generally|accounting (terms|principles)|mutual drafting|drafting|joint (preparation|participation|drafting|negotiation)|no (strict construction|presumption)|strict construction|table of contents|titles|section headings|negotiated agreement|representation by counsel|advice of counsel|independent counsel|ambiguit|language|certain matters of|computation of time|business days?$|time periods|descriptive|including|plural|singular|negotiation of agreement|preparation of (this )?agreement|presumption|review by counsel|legal counsel$|counsel$", "ma.interpretation.rules"),
    (None, r"defin|certain terms|index of|glossary|terms defined|additional terms|specific terms|other terms|cross.?reference|certain (other )?defined|^terms$|capitali[sz]ed terms", "ma.definitions.general"),
    (None, r"recitals|preamble|background|whereas|incorporation of recitals|parties$", "ma.recitals"),
    # ---- fallbacks for representation topics when the article heading gives no context
    (None, r"treasury (shares|stock)|parent.?owned shares|merger shares|the shares$", "ma.consideration.merger_consideration"),
    (None, r"sec (and other )?(reports|documents|filings)|parent sec|^reports$|current report", "ma.rep.sec_reports"),
    (None, r"set.?off", "ma.indemn.setoff"),
    (None, r"independent contractors|works council|service providers|contractors", "ma.rep.labor_employment"),
    (None, r"not to solicit", "ma.restrictive.non_solicit"),
    (None, r"conveyances", "ma.further_assurances"),
    (None, r"recommendation", "ma.nav.deal_protection.recommendation"),
    (None, r"accredited|sophisticat|investment experience|unregistered|economic risk", "ma.rep.investment_intent"),
    (None, r"instruments of (conveyance|transfer)|sellers? certificate|buyers? certificate", "ma.closing.deliverables"),
    (None, r"registration( obligations)?$|rule 144|legend|stand-?off|restricted securities", "ma.consideration.buyer_securities"),
    (None, r"source code", "ma.rep.intellectual_property"),
    (None, r"trade control", "ma.rep.anti_corruption_sanctions"),
    (None, r"cap ?ex", "ma.conduct.restriction.capex"),
    (None, r"balance sheets?", "ma.rep.financial_statements"),
    (None, r"right to control", "ma.conduct.no_control"),
    (None, r"update|breaches", "ma.access.notification"),
    (None, r"customer", "ma.rep.customers_suppliers"),
    (None, r"interim events|since (the )?(reference|balance sheet) date", "ma.rep.absence_of_changes"),
    (None, r"rounding|calculation of (dates|time)|articles and sections|made available|conspicuous|inferences|bold or", "ma.interpretation.rules"),
    (None, r"names? following|sellers.? names?|use of .*names?$", "ma.use_of_name"),
    (None, r"payment of (borrowed|debt)|method of payment", "ma.closing.payments"),
    (None, r"products? liability|warranty obligations", "ma.rep.product_liability_warranty"),
    (None, r"contact with|pcaob", "ma.access.information"),
    (None, r"domestication", "ma.spac.domestication"),
    (None, r"covid|pandemic", "ma.conduct.emergency_exception"),
    (None, r"capital expenditures", "ma.conduct.restriction.capex"),
    (None, r"no (right of )?contribution", "ma.release"),
    (None, r"arms.?length", "ma.interpretation.rules"),
    (None, r"disclosure generally|^disclosures$|data room", "ma.interpretation.disclosure_schedules"),
    (None, r"permitted (liens|encumbrances)", "ma.def.permitted_liens"),
    (None, r"^fraud$", "ma.def.fraud"),
    (None, r"merger of|surviving (corporation|company|entity)|^mergers$|merger closing", "ma.structure.merger"),
    (None, r"^effects$", "ma.structure.effects_of_merger"),
    (None, r"ancillary (agreements|documents)|transaction documents|other agreements|related agreements", "ma.closing.deliverables"),
    (None, r"intellectual property|proprietary rights|patents|trademarks|software", "ma.rep.intellectual_property"),
    (None, r"real (property|estate)|lease|premises|rights-of-way", "ma.rep.real_property"),
    (None, r"environment|hazardous", "ma.rep.environmental"),
    (None, r"legal proceedings|proceedings|^orders?|orders and|claims$", "ma.rep.litigation"),
    (None, r"broker|finder", "ma.rep.brokers"),
    (None, r"subsidiar", "ma.rep.subsidiaries"),
    (None, r"capitali[sz]ation|capital (stock|structure)", "ma.rep.capitalization"),
    (None, r"organi[sz]ation|good standing|existence|qualification|corporate (documents|status)", "ma.rep.organization"),
    (None, r"authori[tz]|enforceab|power$", "ma.rep.authority"),
    (None, r"conflict|contravention|violation|governmental bodies", "ma.rep.no_conflict"),
    (None, r"undisclosed|^liabilities|absence of liabilities", "ma.rep.undisclosed_liabilities"),
    (None, r"absence of|no (material adverse|changes)|business activities; absence", "ma.rep.absence_of_changes"),
    (None, r"sec$|company reports|securities documents|controls and procedures|exchange act", "ma.rep.sec_reports"),
    (None, r"financial (matters|and other statements)|gaap|accounting matters", "ma.rep.financial_statements"),
    (None, r"receivable|inventor|payable", "ma.rep.receivables_inventory"),
    (None, r"bank(ing)? (accounts|relationships)|^accounts$|corporate records|minute", "ma.rep.books_records"),
    (None, r"contracts|commitments", "ma.rep.material_contracts"),
    (None, r"compliance|permits|licenses|regulatory status", "ma.rep.compliance_permits"),
    (None, r"title|sufficiency|personal property|tangible|^properties|^property|lien", "ma.rep.title_sufficiency_assets"),
    (None, r"privacy|data|cyber|it systems|information technology", "ma.rep.privacy_cybersecurity"),
    (None, r"affiliate|related.?party|interested", "ma.rep.affiliate_transactions"),
    (None, r"customers|suppliers", "ma.rep.customers_suppliers"),
    (None, r"solven", "ma.rep.solvency"),
    (None, r"investment (intent|purpose|representation|company)|accredited|securities (act|matters|law)|investor", "ma.rep.investment_intent"),
    (None, r"opinion|fairness", "ma.rep.fairness_opinion"),
    (None, r"information (supplied|to be supplied)", "ma.rep.information_supplied"),
    (None, r"statements true|representations complete|no misrepresentation|complete copies|full disclosure", "ma.rep.full_disclosure"),
    (None, r"disclaimer|no other representation|no additional representation", "ma.rep.no_other_reps"),
    (None, r"independent (evaluation|investigation|review)", "ma.rep.non_reliance"),
    (None, r"(stock )?ownership|^shares$|ownership of", "ma.rep.title_to_shares"),
    (None, r"imbalances|wells|hedging|royalt|hydrocarbon|gas\b|oil\b", "ma.rep.energy_natural_resources"),
    (None, r"investment canada", "ma.regulatory.cfius"),
    (None, r"mutual covenants|affirmative obligations|certain actions|fulfillment of obligations|additional matters", "ma.efforts.general"),
    (None, r"^representations and warranties|^warranties$", "ma.nav.reps"),
    (None, r"business activities|investments$|franchise|spac", "ma.rep.other"),
    # ---- general
    (None, r"notices?$|^notices?|notice$|notices (and|,)|addresses|communications$", "ma.notices"),
    (None, r"entire (agreement|understanding)|complete agreement|integration|whole agreement|prior agreements|entirety|merger clause|sole agreement|supersedes?|^agreement$", "ma.entire_agreement"),
    (None, r"sever|invalid|partial invalidity|enforceability|unenforceab|validity$|separab|illegal|legality", "ma.severability"),
    (None, r"counterpart|facsimile|electronic (signature|delivery|transmission|execution)|execution (of|in|and)|signatures?$|pdf|multiple (originals|copies)|effectiveness$|execution$|duplicate originals|delivery by|originals", "ma.counterparts"),
    (None, r"amend|modif|supplement$|changes$|variation", "ma.amendment"),
    (None, r"waiver|extension$|extension of time|extension;|no (implied )?waiver|failure or (delay|indulgence)|delay|forbearance|extensions$", "ma.waiver"),
    (None, r"assign|successor|binding (effect|agreement|nature|upon)|benefit$|inure|parties bound|binding$|heirs|transfer of (rights|agreement)|delegation|benefits of agreement", "ma.assignment"),
    (None, r"third.?part(y|ies) (beneficiar|rights)|beneficiar|parties in interest|rights of third|no (third|other) (party|parties|person)|no benefit to others|no rights in third|persons (having|bound|benefit)|third part(y|ies)$|benefit of (the )?parties|no third-|contracts \(rights of third", "ma.third_party_beneficiaries"),
    (None, r"expense|fees$|costs|fees and|payment of fees|transaction (costs|fees|expenses)|professional fees|^fees|certain fees", "ma.expenses"),
    (None, r"obligations? of (parent|merger sub|the parent|purchaser|buyer|acquisition|sub|holdco|newco)|merger sub (compliance|obligations|consent|stockholder|approval|vote)|parent (vote|undertaking|consent|approval|agreement|compliance)|actions? of (parent|merger sub|sole)|performance (by|of) (merger sub|parent|sub|purchaser)|compliance by merger sub|sole stockholder|consent of sole|merger sub$|approval (of|by) (sole|parent)|agreements of parent|^parent$|sub (approval|compliance|stockholder)|acquisition sub|joinder|formation of|capacity|company obligations|obligations of the company|several (liability|obligations)|joint and several|parties$|additional parties|liability of|trustee|non-parties", "ma.merger_sub_obligations"),
    (None, r"legal representation|waiver of conflict|conflicts? (and|of|waiver)|privilege|attorney-?client|retention of counsel|representation of|conflict of interest|counsel (conflict|representation)|^conflicts?$|continuing representation|legal counsel|law firm|certain legal", "ma.legal_representation"),
    (None, r"time (is )?of (the )?essence|currency|dollars|time$|performance$|force majeure|spousal|community property|waiver of (appraisal|rights|certain)|exhibits|interest$|business day|payments (in|by)|wire|manner of payment|relationship of (the )?parties|no partnership|no joint venture|non-business|specific disclosures|obligation to|facsimile|construction$|miscellaneous|general$|other provisions|^other$|standard|power of attorney|powers of attorney|electronic|english language|compliance with (laws|law)|effect$|consent$|consents and approvals$|approval$|certain (matters|understandings|agreements|acknowledgements?|provisions)|acknowledg|understanding|company (name|matters)|insurance$|insurance (matters|coverage|policies)|maintenance of insurance|operations|bankruptcy|court approval|sale order|bidding procedures|cure (costs|amounts)|bankruptcy court|plan of arrangement|interim order|final order|court order|^the arrangement|hearing|scheme", "ma.misc.other"),
]

CATCH_ALL = {"ma.other.reserved", "ma.misc.other", "ma.rep.other", "ma.covenant.other", "ma.condition.other",
             "ma.termination.right.other", "ma.other.unclassified", "tax.other.unclassified"}

_COMPILED = [(ctx, re.compile(pattern), target) for ctx, pattern, target in RULES]
_CTX = {name: re.compile(pattern) for name, pattern in CONTEXTS.items()}


def norm_title(text: str | None) -> str:
    if not text or text == "NULL":
        return ""
    text = unicodedata.normalize("NFKC", text).replace("�", "'").replace("\\n", " ").replace("\\t", " ").lower()
    text = re.sub(r"^\s*(section|sec\.?|§+)\s*", "", text)
    text = re.sub(r"^\s*[\(\[]?\d+(\.\d+)*[a-z]?[\)\]]?[.:\-–—]*\s*", "", text)
    text = re.sub(r"^\s*(article|part)\s+[\divxlc]+[.:\-–—]*\s*", "", text)
    text = re.sub(r"[“”\"'’‘`]", "", text)
    return re.sub(r"\s+", " ", text).strip(" .:;,-–—")


def contexts(article: str) -> set[str]:
    article = norm_title(article)
    return {name for name, rx in _CTX.items() if rx.search(article)}


def map_part(part: str, ctx: set[str]) -> str | None:
    for rule_ctx, rx, target in _COMPILED:
        if rule_ctx is not None and rule_ctx not in ctx:
            continue
        if rx.search(part):
            return target
    return None


def classify(title: str, ctx: set[str], nav_ids: set[str]) -> tuple[str, list[str]]:
    if not title:
        return "untitled", []
    parts = [p.strip() for p in re.split(r"\s*;\s*", title) if p.strip()] or [title]
    targets = [map_part(part, ctx) for part in parts]
    hits = [t for t in targets if t]
    if not hits:
        return "unmapped", []
    if len(hits) < len(targets):
        return "partial", hits
    if all(t in CATCH_ALL for t in hits):
        return "catch_all", hits
    if any(t in nav_ids for t in hits):
        return "group", hits
    return "leaf", hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tsv", type=Path)
    parser.add_argument("--unmapped", type=int, default=0, help="print this many random unmapped titles")
    parser.add_argument("--top-unmapped", type=int, default=0, help="print the most frequent unmapped titles")
    parser.add_argument("--by-concept", action="store_true", help="print section and agreement counts per target")
    args = parser.parse_args()

    from etl.domain import taxonomy_pack

    taxonomy = taxonomy_pack.load()
    nav_ids = {node.id for node in taxonomy.nodes.values() if node.is_nav}
    unknown = sorted({target for _, _, target in RULES if target not in taxonomy.nodes})
    if unknown:
        print("Rules name unknown ids:", unknown, file=sys.stderr)
        return 1

    outcome: collections.Counter[str] = collections.Counter()
    by_deal: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    unmapped: collections.Counter[str] = collections.Counter()
    sections_per: collections.Counter[str] = collections.Counter()
    agreements_per: dict[str, set[str]] = collections.defaultdict(set)
    agreements: set[str] = set()
    total = 0
    with args.tsv.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for row in reader:
            if len(row) < 5:
                continue
            agreement, deal_type, target_type, article, section = row[:5]
            total += 1
            agreements.add(agreement)
            title = norm_title(section)
            status, hits = classify(title, contexts(article), nav_ids)
            outcome[status] += 1
            by_deal[f"{deal_type}/{target_type}"][status] += 1
            if status in ("unmapped", "partial"):
                unmapped[title] += 1
            for target in set(hits):
                sections_per[target] += 1
                agreements_per[target].add(agreement)

    order = ["leaf", "group", "partial", "catch_all", "untitled", "unmapped"]
    titled = total - outcome["untitled"]
    print(f"sections: {total:,}   agreements: {len(agreements):,}   titled sections: {titled:,}")
    print(f"{'outcome':<10} {'sections':>10} {'of all':>8} {'of titled':>10}")
    for name in order:
        share_titled = "" if name == "untitled" else f"{outcome[name] / titled:>9.1%}"
        print(f"{name:<10} {outcome[name]:>10,} {outcome[name] / total:>8.1%} {share_titled:>10}")
    print("\nleaf + group share of titled sections, by deal type and target type:")
    for key in sorted(by_deal):
        bucket = by_deal[key]
        n_titled = sum(bucket.values()) - bucket["untitled"]
        if n_titled:
            print(f"  {key:<42} {(bucket['leaf'] + bucket['group']) / n_titled:>6.1%}  (n={n_titled:,})")

    concept_ids = {node.id for node in taxonomy.nodes.values() if node.is_concept}
    never = sorted(concept_ids - set(sections_per))
    print(f"\nconcepts that no title rule reaches (expected for components, definitions and clause-level concepts): {len(never)}")
    if args.by_concept:
        print("\nsections  agreements  target")
        for target, count in sections_per.most_common():
            print(f"{count:>8,}  {len(agreements_per[target]):>10,}  {target}")
        print("\nnever reached by a title rule:")
        for target in never:
            print(f"  {target}")
    if args.top_unmapped:
        print(f"\nmost frequent unmapped or partly mapped titles (distinct: {len(unmapped):,}):")
        for title, count in unmapped.most_common(args.top_unmapped):
            print(f"{count:>7,}  {title}")
    if args.unmapped:
        rng = random.Random(20261122)
        pool = [title for title, count in unmapped.items() for _ in range(count)]
        print(f"\nrandom sample of {args.unmapped} unmapped or partly mapped sections (weighted by frequency):")
        for title in rng.sample(pool, min(args.unmapped, len(pool))):
            print(f"  {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
