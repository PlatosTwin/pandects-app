import os
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, MetaData, Table, func, desc, Column, CHAR, TEXT, Table
from sqlalchemy.dialects.mysql import LONGTEXT, TINYTEXT

# ── Flask setup ──────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": ["http://localhost:8080", "http://127.0.0.1:8080"]}},
    methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL", "mysql+pymysql://root:abotbamxpeh@localhost:3306/mna"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# ── Reflect your existing llm_output table with a standalone engine ────
engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])
metadata = MetaData()

llm_output_table = Table(
    "llm_output",
    metadata,
    autoload_with=engine,
)
prompts_table = Table(
    "prompts",
    metadata,
    autoload_with=engine,
)
agreements_table = Table(
    "agreements",
    metadata,
    autoload_with=engine,
)
xml_table = Table(
    "xml",
    metadata,
    autoload_with=engine,
)
sections_table = Table(
    "sections",
    metadata,
    Column("agreement_uuid", CHAR(36), nullable=False),
    Column("section_uuid",   CHAR(36), primary_key=True),
    Column("article_title",  TEXT,     nullable=False),
    Column("section_title",  TEXT,     nullable=False),
    Column("xml_content",    LONGTEXT, nullable=False),
    Column("article_standard_id",    TINYTEXT, nullable=False),
    Column("section_standard_id",    TINYTEXT, nullable=False),
    schema="mna",
)
taxonomy_table = Table(
    "taxonomy",
    metadata,
    autoload_with=engine,
)


class LLMOut(db.Model):
    __table__ = llm_output_table


class Prompts(db.Model):
    __table__ = prompts_table


class Sections(db.Model):
    __table__ = sections_table


class Agreements(db.Model):
    __table__ = agreements_table


class XML(db.Model):
    __table__ = xml_table
    
    
class Taxonomy(db.Model):
    __table__ = taxonomy_table


@app.route("/api/llm/<string:page_uuid>", methods=["GET"])
def get_llm(page_uuid):
    # pick the most-recent prompt for this page (excluding SKIP outputs)
    latest_prompt_id = (
        db.session.query(Prompts.prompt_id)
        .join(LLMOut, Prompts.prompt_id == LLMOut.prompt_id)
        .filter(LLMOut.page_uuid == page_uuid)
        .filter(func.coalesce(LLMOut.llm_output_corrected, LLMOut.llm_output) != "SKIP")
        .order_by(desc(Prompts.updated_at))
        .limit(1)
        .scalar()
    )
    if not latest_prompt_id:
        abort(404)

    # fetch the LLMOut record
    record = LLMOut.query.get_or_404((page_uuid, latest_prompt_id))

    return jsonify(
        {
            "pageUuid": record.page_uuid,
            "promptId": record.prompt_id,
            "llmOutput": record.llm_output,
            "llmOutputCorrected": record.llm_output_corrected,
        }
    )


@app.route("/api/llm/<string:page_uuid>/<string:prompt_id>", methods=["PUT"])
def update_llm(page_uuid, prompt_id):
    data = request.get_json()
    corrected = data.get("llmOutputCorrected")
    if corrected is None:
        return jsonify({"error": "llmOutputCorrected is required"}), 400

    record = LLMOut.query.get_or_404((page_uuid, prompt_id))
    record.llm_output_corrected = corrected
    db.session.commit()
    return jsonify({"status": "updated"}), 200


@app.route("/api/agreements/<string:agreement_uuid>", methods=["GET"])
def get_agreement(agreement_uuid):
    # query year, target, acquirer, xml, url for this agreement_uuid
    row = (
        db.session.query(
            Agreements.year,
            Agreements.target,
            Agreements.acquirer,
            Agreements.url,
            XML.xml
        )
        .join(XML, XML.agreement_uuid == Agreements.uuid)
        .filter(Agreements.uuid == agreement_uuid)
        .first()
    )

    if row is None:
        abort(404)

    year, target, acquirer, url, xml_content = row
    return jsonify({
        "year":     year,
        "target":   target,
        "acquirer": acquirer,
        "url":      url,
        "xml":      xml_content
    })


@app.route("/api/search", methods=["GET"])
def search_sections():
    # pull in optional query params - now supporting multiple values
    years        = request.args.getlist("year")
    targets      = request.args.getlist("target")
    acquirers    = request.args.getlist("acquirer")
    clause_types = request.args.getlist("clauseType")
    standard_ids = request.args.getlist("standardId")

    # build the base ORM query
    q = (
        db.session
        .query(
            Sections.section_uuid,
            Sections.agreement_uuid,
            Sections.xml_content,
            Sections.article_title,
            Sections.section_title,
            Agreements.acquirer,
            Agreements.target,
            Agreements.year,
        )
        .join(Agreements, Sections.agreement_uuid == Agreements.uuid)
    )

    # apply filters only when provided - now handling multiple values
    if years:
        # Convert years to integers for filtering
        year_ints = [int(year) for year in years if year.isdigit()]
        if year_ints:
            q = q.filter(Agreements.year.in_(year_ints))

    if targets:
        # Use OR conditions for multiple targets with ILIKE
        target_conditions = [Agreements.target.ilike(f"%{target}%") for target in targets]
        q = q.filter(db.or_(*target_conditions))

    if acquirers:
        # Use OR conditions for multiple acquirers with ILIKE
        acquirer_conditions = [Agreements.acquirer.ilike(f"%{acquirer}%") for acquirer in acquirers]
        q = q.filter(db.or_(*acquirer_conditions))

    if standard_ids:
        q = (
            q.join(Taxonomy, Sections.section_standard_id == Taxonomy.standard_id)
             .filter(Taxonomy.type == "section")
             .filter(Taxonomy.standard_id.in_(standard_ids))
        )

    rows = q.all()

    # marshal into JSON
    results = [
        {
            "id":             r.section_uuid,
            "agreementUuid":  r.agreement_uuid,
            "sectionUuid":    r.section_uuid,
            "xml":            r.xml_content,
            "articleTitle":   r.article_title,
            "sectionTitle":   r.section_title,
            "acquirer":       r.acquirer,
            "target":         r.target,
            "year":           r.year,
        }
        for r in rows
    ]

    return jsonify(results), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
