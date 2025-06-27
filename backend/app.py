import os
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, MetaData, Table, func, desc

# ── Flask setup ──────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": "http://localhost:8080"}},
    methods=["GET", "PUT", "OPTIONS"],
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
prompt_table = Table(
    "prompts",
    metadata,
    autoload_with=engine,
)


class LLMOut(db.Model):
    __table__ = llm_output_table


class Prompt(db.Model):
    __table__ = prompt_table


# @app.route('/api/llm/<string:page_uuid>/<string:prompt_id>', methods=['GET'])
# def get_llm(page_uuid, prompt_id):
#     # pass a tuple of primary‐key values to get_or_404
#     p = LLMOut.query.get_or_404((page_uuid, prompt_id))


#     return jsonify({
#         'pageUuid':           p.page_uuid,
#         'promptId':           p.prompt_id,
#         'llmOutput':          p.llm_output,
#         'llmOutputCorrected': p.llm_output_corrected
#     })
@app.route("/api/llm/<string:page_uuid>", methods=["GET"])
def get_llm(page_uuid):
    # pick the most-recent prompt for this page (excluding SKIP outputs)
    latest_prompt_id = (
        db.session.query(Prompt.prompt_id)
        .join(LLMOut, Prompt.prompt_id == LLMOut.prompt_id)
        .filter(LLMOut.page_uuid == page_uuid)
        .filter(func.coalesce(LLMOut.llm_output_corrected, LLMOut.llm_output) != "SKIP")
        .order_by(desc(Prompt.updated_at))
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
