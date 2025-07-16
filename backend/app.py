import os
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    func,
    desc,
    Column,
    CHAR,
    TEXT,
    Table,
    text,
)
from sqlalchemy.dialects.mysql import LONGTEXT, TINYTEXT
from dotenv import load_dotenv

load_dotenv()

# ── Flask setup ────────────────────────────��─────────────────────────────
app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:8080",
                "http://127.0.0.1:8080",
                "https://pandects-app.fly.dev",
            ]
        }
    },
    methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

DB_USER = os.environ["MARIADB_USER"]
DB_PASS = os.environ["MARIADB_PASSWORD"]
DB_HOST = os.environ["MARIADB_HOST"]
DB_NAME = os.environ["MARIADB_DATABASE"]

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"
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
    Column("section_uuid", CHAR(36), primary_key=True),
    Column("article_title", TEXT, nullable=False),
    Column("section_title", TEXT, nullable=False),
    Column("xml_content", LONGTEXT, nullable=False),
    Column("article_standard_id", TINYTEXT, nullable=False),
    Column("section_standard_id", TINYTEXT, nullable=False),
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
            XML.xml,
        )
        .join(XML, XML.agreement_uuid == Agreements.uuid)
        .filter(Agreements.uuid == agreement_uuid)
        .first()
    )

    if row is None:
        abort(404)

    year, target, acquirer, url, xml_content = row
    return jsonify(
        {
            "year": year,
            "target": target,
            "acquirer": acquirer,
            "url": url,
            "xml": xml_content,
        }
    )


@app.route("/api/filter-options", methods=["GET"])
def get_filter_options():
    """Fetch distinct targets and acquirers from the database"""
    try:
        # Execute the SQL query to get distinct targets and acquirers
        result = db.session.execute(
            text(
                """
            SELECT DISTINCT target, acquirer
            FROM mna.agreements a
            JOIN mna.xml x ON a.uuid = x.agreement_uuid
            ORDER BY target, acquirer
            """
            )
        ).fetchall()

        # Extract unique targets and acquirers
        targets = sorted(set(row[0] for row in result if row[0]))
        acquirers = sorted(set(row[1] for row in result if row[1]))

        return jsonify({"targets": targets, "acquirers": acquirers}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["GET"])
def search_sections():
    # pull in optional query params - now supporting multiple values
    years = request.args.getlist("year")
    targets = request.args.getlist("target")
    acquirers = request.args.getlist("acquirer")
    clause_types = request.args.getlist("clauseType")
    standard_ids = request.args.getlist("standardId")
    transaction_sizes = request.args.getlist("transactionSize")
    transaction_types = request.args.getlist("transactionType")
    consideration_types = request.args.getlist("considerationType")
    target_types = request.args.getlist("targetType")

    # pagination parameters
    page = request.args.get("page", 1, type=int)
    page_size = request.args.get("pageSize", 25, type=int)

    # Validate pagination parameters
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:  # Cap max page size for performance
        page_size = 25

    # build the base ORM query
    q = db.session.query(
        Sections.section_uuid,
        Sections.agreement_uuid,
        Sections.xml_content,
        Sections.article_title,
        Sections.section_title,
        Agreements.acquirer,
        Agreements.target,
        Agreements.year,
    ).join(Agreements, Sections.agreement_uuid == Agreements.uuid)

    # apply filters only when provided - now handling multiple values
    if years:
        # Convert years to integers for filtering
        year_ints = [int(year) for year in years if year.isdigit()]
        if year_ints:
            q = q.filter(Agreements.year.in_(year_ints))

    if targets:
        # Use OR conditions for multiple targets with ILIKE
        target_conditions = [
            Agreements.target.ilike(f"%{target}%") for target in targets
        ]
        q = q.filter(db.or_(*target_conditions))

    if acquirers:
        # Use OR conditions for multiple acquirers with ILIKE
        acquirer_conditions = [
            Agreements.acquirer.ilike(f"%{acquirer}%") for acquirer in acquirers
        ]
        q = q.filter(db.or_(*acquirer_conditions))

    if standard_ids:
        q = (
            q.join(Taxonomy, Sections.section_standard_id == Taxonomy.standard_id)
            .filter(Taxonomy.type == "section")
            .filter(Taxonomy.standard_id.in_(standard_ids))
        )

    # Transaction Size filter - convert ranges to DB values
    if transaction_sizes:
        size_conditions = []
        for size_range in transaction_sizes:
            if size_range == "100M - 250M":
                size_conditions.append(
                    db.and_(
                        Agreements.transaction_size >= 100000000,
                        Agreements.transaction_size < 250000000,
                    )
                )
            elif size_range == "250M - 500M":
                size_conditions.append(
                    db.and_(
                        Agreements.transaction_size >= 250000000,
                        Agreements.transaction_size < 500000000,
                    )
                )
            elif size_range == "500M - 750M":
                size_conditions.append(
                    db.and_(
                        Agreements.transaction_size >= 500000000,
                        Agreements.transaction_size < 750000000,
                    )
                )
            elif size_range == "750M - 1B":
                size_conditions.append(
                    db.and_(
                        Agreements.transaction_size >= 750000000,
                        Agreements.transaction_size < 1000000000,
                    )
                )
            elif size_range == "1B - 5B":
                size_conditions.append(
                    db.and_(
                        Agreements.transaction_size >= 1000000000,
                        Agreements.transaction_size < 5000000000,
                    )
                )
            elif size_range == "5B - 10B":
                size_conditions.append(
                    db.and_(
                        Agreements.transaction_size >= 5000000000,
                        Agreements.transaction_size < 10000000000,
                    )
                )
            elif size_range == "10B - 20B":
                size_conditions.append(
                    db.and_(
                        Agreements.transaction_size >= 10000000000,
                        Agreements.transaction_size < 20000000000,
                    )
                )
            elif size_range == "20B+":
                size_conditions.append(Agreements.transaction_size >= 20000000000)
        if size_conditions:
            q = q.filter(db.or_(*size_conditions))

    # Transaction Type filter
    if transaction_types:
        # Convert frontend values to DB enum values
        db_transaction_types = []
        for t_type in transaction_types:
            if t_type == "Strategic":
                db_transaction_types.append("strategic")
            elif t_type == "Financial":
                db_transaction_types.append("financial")
        if db_transaction_types:
            q = q.filter(Agreements.transaction_type.in_(db_transaction_types))

    # Consideration Type filter
    if consideration_types:
        # Convert frontend values to DB enum values
        db_consideration_types = []
        for c_type in consideration_types:
            if c_type == "All stock":
                db_consideration_types.append("stock")
            elif c_type == "All cash":
                db_consideration_types.append("cash")
            elif c_type == "Mixed":
                db_consideration_types.append("mixed")
        if db_consideration_types:
            q = q.filter(Agreements.consideration_type.in_(db_consideration_types))

    # Target Type filter
    if target_types:
        # Convert frontend values to DB enum values
        db_target_types = []
        for t_type in target_types:
            if t_type == "Public":
                db_target_types.append("public")
            elif t_type == "Private":
                db_target_types.append("private")
        if db_target_types:
            q = q.filter(Agreements.target_type.in_(db_target_types))

    # Use SQLAlchemy's paginate() method
    try:
        paginated = q.paginate(
            page=page,
            per_page=page_size,
            error_out=False
        )
    except Exception as e:
        return jsonify({"error": f"Pagination error: {str(e)}"}), 400

    # marshal into JSON with pagination metadata
    results = [
        {
            "id": r.section_uuid,
            "agreementUuid": r.agreement_uuid,
            "sectionUuid": r.section_uuid,
            "xml": r.xml_content,
            "articleTitle": r.article_title,
            "sectionTitle": r.section_title,
            "acquirer": r.acquirer,
            "target": r.target,
            "year": r.year,
        }
        for r in paginated.items
    ]

    # Return results with pagination metadata
    return jsonify({
        "results": results,
        "page": paginated.page,
        "pageSize": paginated.per_page,
        "totalCount": paginated.total,
        "totalPages": paginated.pages,
        "hasNext": paginated.has_next,
        "hasPrev": paginated.has_prev,
        "nextNum": paginated.next_num,
        "prevNum": paginated.prev_num
    }), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
