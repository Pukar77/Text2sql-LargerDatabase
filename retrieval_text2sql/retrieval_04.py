from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os
import json
import cx_Oracle
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from decimal import Decimal
from datetime import date, datetime

router = APIRouter()


class UserQuestion(BaseModel):
    question: str


# =====================================================
# 1. ENVIRONMENT CONFIGURATION
# =====================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


# =====================================================
# 2. EMBEDDING MODEL CONFIGURATION
# =====================================================

MODEL_NAME = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(MODEL_NAME)


# =====================================================
# 3. QDRANT CONFIGURATION
# =====================================================

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "oracle_schema_tables"

qdrant_client = QdrantClient(
    url=QDRANT_URL
)


# =====================================================
# 4. GROQ CLIENT CONFIGURATION
# =====================================================

client = OpenAI(
    api_key=os.getenv("api_key"),
    base_url="https://api.groq.com/openai/v1",
)


# =====================================================
# 12. LOCAL RESULT PREPROCESSING FUNCTION
# =====================================================

def format_value(value):
    """
    Format Oracle values into readable output.
    """

    if value is None:
        return "N/A"

    if isinstance(value, Decimal):
        return f"{value:,.2f}"

    if isinstance(value, float):
        return f"{value:,.2f}"

    if isinstance(value, int):
        return f"{value:,}"

    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")

    return str(value)


def generate_local_answer(user_question, results):
    """
    Generate a natural-language answer locally
    without making another LLM API call.
    """

    if not results:

        return "No matching data was found."

    # =================================================
    # CASE 1: ONLY ONE RECORD RETURNED
    # =================================================

    if len(results) == 1:

        row = results[0]

        # Only one column returned
        if len(row) == 1:

            column_name = list(row.keys())[0]

            value = row[column_name]

            readable_column = (
                column_name
                .replace("_", " ")
                .title()
            )

            return (
                f"The result for {readable_column} is "
                f"{format_value(value)}."
            )

        # Multiple columns returned
        answer_parts = []

        for column_name, value in row.items():

            readable_column = (
                column_name
                .replace("_", " ")
                .title()
            )

            answer_parts.append(
                f"{readable_column}: "
                f"{format_value(value)}"
            )

        return (
            "The result is: "
            + ", ".join(answer_parts)
            + "."
        )

    # =================================================
    # CASE 2: MULTIPLE RECORDS RETURNED
    # =================================================

    answer_parts = []

    for index, row in enumerate(results[:20], start=1):

        row_parts = []

        for column_name, value in row.items():

            readable_column = (
                column_name
                .replace("_", " ")
                .title()
            )

            row_parts.append(
                f"{readable_column}: "
                f"{format_value(value)}"
            )

        answer_parts.append(
            f"{index}. "
            + ", ".join(row_parts)
        )

    answer = (
        f"Found {len(results)} records. "
        + " ".join(answer_parts)
    )

    if len(results) > 20:

        answer += (
            f" Showing the first 20 of "
            f"{len(results)} records."
        )

    return answer


# =====================================================
# 5. FASTAPI ENDPOINT
# =====================================================

@router.post("/ask")
def ask_question(request: UserQuestion):

    user_question = request.question


    # =====================================================
    # 6. RETRIEVE RELEVANT SCHEMA CHUNKS
    # =====================================================

    print("\nRetrieving relevant database schema...")

    # Generate embedding for user question
    query_embedding = embedding_model.encode(
        user_question,
        normalize_embeddings=True
    )

    # Search relevant tables in Qdrant
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=2,
        with_payload=True
    ).points


    if not search_results:

        raise RuntimeError(
            "No relevant database schema found in Qdrant."
        )


    # Extract retrieved table chunks
    retrieved_chunks = []

    print("\n========== RETRIEVED TABLES ==========")

    for result in search_results:

        payload = result.payload

        table_name = payload.get("table_name", "Unknown")

        content = payload.get("content", "")

        print(
            f"Table: {table_name} | "
            f"Similarity Score: {result.score:.4f}"
        )

        if content:
            retrieved_chunks.append(content)


    if not retrieved_chunks:

        raise RuntimeError(
            "Retrieved schema records do not contain table content."
        )


    # Combine retrieved chunks into one schema context
    retrieved_schema = "\n\n".join(retrieved_chunks)


    # =====================================================
    # 7. SYSTEM PROMPT
    # =====================================================

    prompt = f"""
You are an expert Oracle SQL developer.

Convert the user's natural language question into
a valid Oracle SQL query.

DATABASE: Oracle 11g

SCHEMA:
{retrieved_schema}

RULES:

Use only the provided tables and columns.

Use correct JOIN relationships when necessary.

Whenever performing a JOIN operation between tables that contain a COMPANY_CODE column, always include COMPANY_CODE in the JOIN condition to ensure that data is joined only within the same company. For example, i.COMPANY_CODE = s.COMPANY_CODE. Never join records across different companies.

Generate Oracle-compatible SQL.

Never invent table or column names.

Generate SELECT queries only.

Do not generate INSERT, UPDATE, DELETE, DROP,
ALTER, TRUNCATE, or other modification statements.

Do not use SELECT *.

Select only the columns relevant to the user's question.

Avoid selecting unnecessary columns.

For general detail/list requests, select relevant
business columns rather than technical metadata columns.

Limit general detail queries to 50 rows using
Oracle-compatible ROWNUM.

The generated SQL must be executable in Oracle 11g.

Do not include Markdown code fences or explanatory text.

Return the SQL query using the JSON structure
defined by the API response schema.

CRITICAL ORACLE 11g CONSTRAINTS:

NO LIMIT/OFFSET:
Do not use LIMIT or OFFSET.
Use subqueries with ROWNUM or ROW_NUMBER() for pagination.

NO IDENTITY COLUMNS:
Do not use AUTO_INCREMENT.

NO BOOLEAN TYPE:
Use CHAR(1) or NUMBER(1) for boolean flags.

NO IF NOT EXISTS:
Write standard CREATE TABLE statements when applicable.

STRING CONCATENATION:
Use 'col1' || 'col2', never multi-argument CONCAT().

DUAL TABLE:
Append FROM DUAL when selecting expressions or
system dates without a physical table.

MANDATORY SUBQUERY ALIASES:
Every subquery or inline view used in a FROM clause
MUST have an explicit alias.

Example:
SELECT sub.column_name
FROM (
SELECT column_name
FROM table_name
) sub;

STRICT COLUMN SCOPING:
Table aliases defined inside a subquery are scoped
only within that subquery.

The outer query must reference projected columns
through the subquery alias.

RANKED QUERIES:
Never use WHERE ROWNUM = N (where N > 1).

Always use ROW_NUMBER() inside a subquery for
ranked lookups.

Example:

SELECT sub.*
FROM (
SELECT
column_name,
SUM(metric) AS total,
ROW_NUMBER() OVER (
ORDER BY SUM(metric) DESC
) AS rn
FROM table_name
GROUP BY column_name
) sub
WHERE sub.rn = 2;

HIGHEST/LOWEST SALES:
When finding the highest or lowest sales, aggregate
the appropriate sales metric and rank the results
in descending or ascending order respectively.

DATE FILTERING:
For current-year questions, use Oracle-compatible
date filtering based on SYSDATE.


SALES PRICE COLUMN PRIORITY:

When a table contains both TOTAL_PRICE and CALC_TOTAL_PRICE
columns, always use CALC_TOTAL_PRICE for sales calculations
unless the user explicitly specifies otherwise.

CALC_TOTAL_PRICE represents the final sales price after
applying discounts.

TOTAL_PRICE represents the original sales price before
applying discounts.

Therefore:

- For total sales calculations, use SUM(CALC_TOTAL_PRICE).
- For highest or lowest sales calculations, aggregate
  CALC_TOTAL_PRICE.
- For monthly, yearly, brand-wise, category-wise, or
  other sales aggregations, use CALC_TOTAL_PRICE.
- Never use TOTAL_PRICE for sales calculations when
  CALC_TOTAL_PRICE is available in the same table,
  unless the user explicitly asks for the price before
  discounts or specifically mentions TOTAL_PRICE.

This rule takes priority over any default sales metric
selection logic.

"""


    # =====================================================
    # 8. GENERATE SQL USING JSON OBJECT MODE
    # =====================================================

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": (
                        "USER QUESTION:\n"
                        + user_question
                        + "\n\n"
                        "Return a valid JSON object in this exact format:\n"
                        '{"sql": "your Oracle SQL query"}\n\n'
                        "The sql value must be a string containing "
                        "only an Oracle SELECT query. "
                        "Do not include Markdown or explanations."
                    )
                }
            ],

            temperature=0,

            response_format={
                "type": "json_object"
            },

            max_tokens=2048
        )

    except Exception as e:

        print("\n========== GROQ API ERROR ==========")
        print(str(e))

        if hasattr(e, "response") and e.response is not None:
            print("\nResponse body:")
            print(e.response.text)

        raise RuntimeError(
            f"Groq API request failed: {str(e)}"
        ) from e


    # =====================================================
    # 9. INSPECT LLM RESPONSE
    # =====================================================

    print("\n========== RESPONSE ==========")

    response_content = response.choices[0].message.content

    print("\n========== RAW LLM RESPONSE ==========")
    print(repr(response_content))


    if not response_content or not response_content.strip():

        raise RuntimeError(
            "LLM returned an empty response."
        )


    # =====================================================
    # 10. PARSE JSON RESPONSE
    # =====================================================

    try:

        response_data = json.loads(response_content)

    except json.JSONDecodeError as e:

        print("\nInvalid JSON response:")
        print(response_content)

        raise RuntimeError(
            f"Failed to parse LLM JSON response: {str(e)}"
        ) from e


    if "sql" not in response_data:

        raise RuntimeError(
            "LLM response does not contain the 'sql' field."
        )


    sql_query = response_data["sql"].strip()
    sql_query = sql_query.rstrip(";").strip()


    if not sql_query:

        raise RuntimeError(
            "Generated SQL query is empty."
        )


    print("\n========== GENERATED SQL ==========")
    print(sql_query)


    # =====================================================
    # 11. BASIC SQL SAFETY VALIDATION
    # =====================================================

    normalized_sql = sql_query.lstrip()

    if not (
        normalized_sql.upper().startswith("SELECT")
        or normalized_sql.upper().startswith("WITH")
    ):

        raise ValueError(
            "Unsafe SQL rejected. Only SELECT queries are allowed."
        )


    if ";" in sql_query:

        raise ValueError(
            "Multiple SQL statements or statement terminators are not allowed."
        )


    # =====================================================
    # 13. ORACLE DATABASE CONNECTION AND EXECUTION
    # =====================================================

    connection = None
    cursor = None

    try:

        connection = cx_Oracle.connect(
            user=os.getenv("DB_USERNAME"),
            password=os.getenv("DB_PASSWORD"),
            dsn=os.getenv("DB_DSN")
        )

        cursor = connection.cursor()

        print("\nExecuting SQL query...")

        cursor.execute(sql_query)

        # Get column names
        columns = [
            col[0]
            for col in cursor.description
        ]

        # Fetch database records ONCE
        rows = cursor.fetchall()

        # =================================================
        # 14. CONVERT SQL RESULTS INTO JSON
        # =================================================

        results = [
            dict(zip(columns, row))
            for row in rows
        ]

        print("\n========== DATABASE RESULTS ==========")

        print(
            json.dumps(
                results,
                indent=4,
                default=str
            )
        )

        # =================================================
        # 15. GENERATE NATURAL LANGUAGE ANSWER LOCALLY
        # =================================================

        final_answer = generate_local_answer(
            user_question=user_question,
            results=results
        )

        print("\n========== FINAL ANSWER ==========")

        print(
            json.dumps(
                {"answer": final_answer},
                indent=4,
                ensure_ascii=False
            )
        )


    except cx_Oracle.DatabaseError as e:

        error, = e.args

        print("\nOracle Database Error:")
        print(error.message)

        raise

    except Exception as e:

        print("\nProcessing Error:")
        print(str(e))

        raise

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

        print("\nDatabase connection closed.")


    # =====================================================
    # 16. API RESPONSE
    # =====================================================

    return {
        "question": user_question,
        "sql": sql_query,
        "results": results,
        "answer": final_answer
    }