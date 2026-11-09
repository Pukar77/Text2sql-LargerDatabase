
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "schema.json"
OUTPUT_FILE = BASE_DIR / "enriched_schema.json"

client = OpenAI(
    api_key=os.getenv("api_key"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"


def generate_table_description(table):
    """
    Send only one table's metadata to the LLM.
    The LLM returns table and column descriptions.
    """

    table_metadata = {
        "owner": table["owner"],
        "table_name": table["table_name"],
        "table_description": table.get("table_description"),
        "columns": [
            {
                "column_name": col["column_name"],
                "data_type": col["data_type"],
                "data_length": col.get("data_length"),
                "data_precision": col.get("data_precision"),
                "data_scale": col.get("data_scale"),
                "nullable": col.get("nullable"),
                "column_description": col.get("column_description"),
                "key_type": col.get("key_type"),
                "foreign_key": col.get("foreign_key"),
            }
            for col in table["columns"]
        ]
    }


    prompt = f"""
You are an expert Oracle ERP database analyst and
database schema documentation specialist.

Your task is to generate meaningful business-oriented
descriptions for the provided Oracle database table
and all its columns.

The descriptions will be used in a Text-to-SQL system
with semantic search using Sentence Transformers.

Your descriptions must help an LLM understand the
business meaning of tables and columns when generating
Oracle SQL queries.

RULES:

1. TABLE DESCRIPTION:
   - Generate a concise, meaningful description of the
     business purpose of the table.
   - Consider the table name, existing table comments,
     column names, datatypes, and relationships.
   - Describe what kind of business information the
     table stores.
   - Avoid generic descriptions such as "This table
     contains data".

2. CONTEXT-AWARE COLUMN DESCRIPTIONS:
   - Generate a meaningful description for every column.
   - NEVER describe a column in isolation.
   - Always describe each column in the context of its
     parent table.
   - Consider the table name, table description,
     column name, datatype, key information, and other
     columns belonging to the same table.
   - The description must explain the role of the column
     within the business purpose of its parent table.
   - Do not blindly append the table name to the
     description. Generate a natural and meaningful
     business description.

3. AMBIGUOUS COLUMN NAMES:
   - Many Oracle ERP tables contain generic column names
     such as DATE, CODE, NAME, STATUS, DESCRIPTION,
     COMPANY_CODE, CREATED_BY, and AMOUNT.
   - These columns may have different meanings in
     different tables.
   - Infer their contextual meaning using the parent
     table name and available metadata.
   - Never assume that columns with identical names
     have identical business meanings.

   Examples:

   Table: SALES_DETAILS
   Column: DATE
   Description: Date on which the sales transaction
   occurred.

   Table: ITEM_DETAILS
   Column: DATE
   Description: Date associated with the item master
   record.

   Table: EMPLOYEE_DETAILS
   Column: DATE
   Description: Date associated with the employee
   record.

   Table: SALES_DETAILS
   Column: COMPANY_CODE
   Description: Identifier of the company associated
   with the sales invoice transaction.

   Table: ITEM_DETAILS
   Column: COMPANY_CODE
   Description: Identifier of the company associated
   with the item master record.

   Table: SALES_DETAILS
   Column: ITEM_CODE
   Description: Identifier of the item sold in the
   sales invoice transaction.

   Table: ITEM_DETAILS
   Column: ITEM_CODE
   Description: Unique identifier of the item in the
   item master record.

4. USE AVAILABLE METADATA:
   - Use existing table descriptions and column
     descriptions whenever available.
   - Consider datatypes when interpreting columns.
   - Use primary key information to understand
     identifiers of records.
   - Use foreign key information to understand
     references to other tables.
   - Consider related columns within the same table
     to infer business meaning.
   - Do not ignore any provided metadata.

5. ACCURACY AND HALLUCINATION PREVENTION:
   - Use only the supplied metadata to infer meanings.
   - Do not invent business functionality.
   - Do not assume undocumented business rules.
   - If the exact meaning cannot be determined, provide
     a cautious description based on the table context.
   - If the meaning is completely unclear, return
     "UNKNOWN".
   - Do not claim that a column represents a specific
     business event unless the metadata supports it.

6. DESCRIPTION QUALITY:
   - Keep table descriptions concise and informative.
   - Keep column descriptions concise and informative.
   - Prefer natural business language.
   - Avoid unnecessary technical explanations.
   - Avoid repeating the same generic description for
     every column.
   - Descriptions should be useful for semantic search
     and Text-to-SQL generation.

7. PRESERVE ORIGINAL METADATA:
   - Do not modify table names.
   - Do not modify column names.
   - Do not modify datatypes.
   - Do not remove columns.
   - Do not add columns that do not exist.
   - Generate descriptions only.

8. OUTPUT REQUIREMENTS:
   - Return valid JSON only.
   - Do not include Markdown.
   - Do not include explanations outside the JSON.
   - Include every original column exactly once.
   - Preserve the exact spelling of table and column names.
   - Follow the specified JSON structure.

OUTPUT FORMAT:

{{
    "table_name": "TABLE_NAME",
    "table_description": "Business description of the table",
    "columns": [
        {{
            "column_name": "COLUMN_NAME",
            "column_description": "Context-aware business description"
        }}
    ]
}}

DATABASE TABLE METADATA:

{json.dumps(table_metadata, indent=2, ensure_ascii=False)}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    result = json.loads(
        response.choices[0].message.content
    )

    return result


def enrich_table(table):
    """
    Enrich one table without modifying its original metadata.
    """

    result = generate_table_description(table)

    if result["table_name"] != table["table_name"]:
        raise ValueError("LLM returned incorrect table name")

    # Update table description
    table["table_description"] = result["table_description"]

    # Create a mapping of column names to descriptions
    column_descriptions = {
        col["column_name"]: col["column_description"]
        for col in result["columns"]
    }

    # Validate that every original column has a description
    original_columns = {
        col["column_name"] for col in table["columns"]
    }

    generated_columns = set(column_descriptions.keys())

    if original_columns != generated_columns:
        missing = original_columns - generated_columns
        extra = generated_columns - original_columns

        raise ValueError(
            f"Column mismatch. Missing: {missing}, Extra: {extra}"
        )

    # Update only column_description
    for column in table["columns"]:
        column["column_description"] = (
            column_descriptions[column["column_name"]]
        )

    return table


def save_json(data):
    """
    Save JSON incrementally using a temporary file.
    """

    temp_file = OUTPUT_FILE.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    temp_file.replace(OUTPUT_FILE)


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        schema_data = json.load(f)

    tables = schema_data["tables"]

    total_tables = len(tables)

    print(f"Total tables: {total_tables}")

    # Save initial output structure
    save_json(schema_data)

    for index, table in enumerate(tables, start=1):

        table_name = table["table_name"]

        print(
            f"\n[{index}/{total_tables}] "
            f"Processing: {table_name}"
        )

        # Skip already enriched tables
        if (
            table.get("table_description")
            and all(
                col.get("column_description")
                for col in table["columns"]
            )
        ):
            print("Already enriched. Skipping.")
            continue

        try:

            enriched_table = enrich_table(table)

            # Update schema with enriched table
            schema_data["tables"][index - 1] = enriched_table

            # Save immediately after each table
            save_json(schema_data)

            print("Successfully enriched.")

        except Exception as e:

            print(f"Error processing {table_name}: {e}")

            # Original metadata is preserved
            # Continue to the next table

        time.sleep(1)

    print("\nSchema enrichment completed.")

    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


#After running this file, we will get a file named as enriched_schema.json which will have the schema of the database in json format with table and column descriptions.
