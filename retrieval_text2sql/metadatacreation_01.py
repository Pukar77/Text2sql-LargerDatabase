
import json
import os
from pathlib import Path
from collections import OrderedDict
from decimal import Decimal

import cx_Oracle
from dotenv import load_dotenv


# ==================================================
# 1. ENVIRONMENT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DB_USER = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DSN = os.getenv("DB_DSN")

INSTANT_CLIENT_DIR = BASE_DIR / "instantclient_21_11"

SQL_FILE = BASE_DIR / "schema_query.sql"
OUTPUT_FILE = BASE_DIR / "schema.json"


# ==================================================
# 2. INITIALIZE ORACLE INSTANT CLIENT
# ==================================================

def initialize_oracle_client():

    if not INSTANT_CLIENT_DIR.exists():

        raise FileNotFoundError(
            f"Instant Client not found: {INSTANT_CLIENT_DIR}"
        )

    try:

        cx_Oracle.init_oracle_client(
            lib_dir=str(INSTANT_CLIENT_DIR)
        )

        print("Oracle Instant Client initialized successfully.")

    except cx_Oracle.Error as error:

        print(f"Oracle Client initialization error: {error}")
        raise


# ==================================================
# 3. VALIDATE ENVIRONMENT CONFIGURATION
# ==================================================

def validate_configuration():

    missing = []

    if not DB_USER:
        missing.append("ORACLE_USER")

    if not DB_PASSWORD:
        missing.append("ORACLE_PASSWORD")

    if not DB_DSN:
        missing.append("ORACLE_DSN")

    if missing:

        raise ValueError(
            f"Missing environment variables: {', '.join(missing)}"
        )


# ==================================================
# 4. LOAD SQL QUERY
# ==================================================

def load_sql_query():

    if not SQL_FILE.exists():

        raise FileNotFoundError(
            f"SQL file not found: {SQL_FILE}"
        )

    with open(SQL_FILE, "r", encoding="utf-8") as file:

        return file.read()


# ==================================================
# 5. JSON SERIALIZER
# ==================================================

def json_serializer(value):

    if isinstance(value, Decimal):
        return float(value)

    if hasattr(value, "isoformat"):
        return value.isoformat()

    raise TypeError(
        f"Type {type(value)} is not JSON serializable"
    )


# ==================================================
# 6. EXECUTE SCHEMA METADATA QUERY
# ==================================================

def fetch_schema_metadata():

    connection = None
    cursor = None

    try:

        print("\nConnecting to Oracle Database...")

        connection = cx_Oracle.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            dsn=DB_DSN,
            encoding="UTF-8"
        )

        print("Connected to Oracle Database successfully.")

        cursor = connection.cursor()

        query = load_sql_query()

        print("\nExecuting schema metadata query...")

        cursor.execute(query)

        column_names = [
            column[0] for column in cursor.description
        ]

        rows = cursor.fetchall()

        print(f"Fetched {len(rows)} metadata rows.")

        metadata = []

        for row in rows:

            record = dict(zip(column_names, row))

            metadata.append(record)

        return metadata

    except cx_Oracle.Error as error:

        print(f"Oracle Database Error: {error}")
        raise

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()

            print("Oracle connection closed.")


# ==================================================
# 7. REMOVE COMPLETELY EMPTY COLUMNS
# ==================================================

def filter_empty_columns(metadata):

    connection = None
    cursor = None

    try:

        connection = cx_Oracle.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            dsn=DB_DSN,
            encoding="UTF-8"
        )

        cursor = connection.cursor()

        # Group columns by owner and table
        tables = OrderedDict()

        for record in metadata:

            table_key = (
                record["OWNER"],
                record["TABLE_NAME"]
            )

            if table_key not in tables:

                tables[table_key] = []

            tables[table_key].append(record)

        filtered_metadata = []

        # Oracle data types that cannot be used with COUNT()
        unsupported_types = {
            "LONG",
            "LONG RAW"
        }

        for (owner, table_name), columns in tables.items():

            print(
                f"\nChecking table: {owner}.{table_name}"
            )

            # Quote identifiers safely
            def quote_identifier(name):

                return '"' + name.replace('"', '""') + '"'

            qualified_table = (
                f"{quote_identifier(owner)}."
                f"{quote_identifier(table_name)}"
            )

            # Separate countable and unsupported columns
            countable_columns = [
                col for col in columns
                if col["DATA_TYPE"].upper() not in unsupported_types
            ]

            # Retain unsupported types conservatively
            valid_columns = [
                col for col in columns
                if col["DATA_TYPE"].upper() in unsupported_types
            ]

            # Build COUNT expressions
            count_expressions = [
                (
                    f'COUNT({quote_identifier(col["COLUMN_NAME"])}) '
                    f'AS C{i}'
                )
                for i, col in enumerate(countable_columns)
            ]

            # Execute one aggregate query per table
            sql = f"""
                SELECT
                    COUNT(*) AS TOTAL_ROWS
                    {", " + ", ".join(count_expressions) if count_expressions else ""}
                FROM {qualified_table}
            """

            try:

                cursor.execute(sql)

                result = cursor.fetchone()

                total_rows = result[0]

                # Empty table: retain all columns
                if total_rows == 0:

                    filtered_metadata.extend(columns)

                    print(
                        "Table contains no rows. "
                        "Retaining all columns."
                    )

                    continue

                non_null_counts = result[1:]

                # Check every countable column
                for col, count in zip(
                    countable_columns,
                    non_null_counts
                ):

                    column_name = col["COLUMN_NAME"]

                    if count > 0:

                        valid_columns.append(col)

                        print(
                            f"  KEEP   : {column_name} "
                            f"({count}/{total_rows} populated)"
                        )

                    else:

                        print(
                            f"  REMOVE : {column_name} "
                            "(completely NULL)"
                        )

                # Preserve original column ordering
                valid_names = {
                    col["COLUMN_NAME"]
                    for col in valid_columns
                }

                valid_columns = [
                    col for col in columns
                    if col["COLUMN_NAME"] in valid_names
                ]

                filtered_metadata.extend(valid_columns)

                removed_count = (
                    len(columns) - len(valid_columns)
                )

                print(
                    f"Table summary: "
                    f"{len(valid_columns)} retained, "
                    f"{removed_count} removed."
                )

            except cx_Oracle.Error as error:

                # If checking fails, retain all columns
                print(
                    f"Could not check table {owner}.{table_name}: "
                    f"{error}"
                )

                print("Retaining all columns for safety.")

                filtered_metadata.extend(columns)

        print("\nEmpty column filtering completed.")

        print(
            f"Original columns: {len(metadata)}"
        )

        print(
            f"Remaining columns: {len(filtered_metadata)}"
        )

        print(
            f"Removed columns: "
            f"{len(metadata) - len(filtered_metadata)}"
        )

        return filtered_metadata

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ==================================================
# 8. CONVERT METADATA INTO STRUCTURED JSON
# ==================================================

def build_schema_json(metadata):

    tables = OrderedDict()

    for record in metadata:

        owner = record["OWNER"]
        table_name = record["TABLE_NAME"]

        table_key = f"{owner}.{table_name}"

        # Initialize table
        if table_key not in tables:

            tables[table_key] = {

                "owner": owner,

                "table_name": table_name,

                "table_description": record.get(
                    "TABLE_DESCRIPTION"
                ),

                "columns": []
            }

        # Column information
        column = {

            "column_name": record["COLUMN_NAME"],

            "data_type": record["DATA_TYPE"],

            "data_length": record["DATA_LENGTH"],

            "data_precision": record["DATA_PRECISION"],

            "data_scale": record["DATA_SCALE"],

            "nullable": record["NULLABLE"],

            "column_id": record["COLUMN_ID"],

            "column_description": record.get(
                "COLUMN_DESCRIPTION"
            ),

            "key_type": record["KEY_TYPE"],

            "primary_key_constraint": record.get(
                "PK_CONSTRAINT_NAME"
            ),

            "foreign_key": None,

            "unique_key_constraint": record.get(
                "UK_CONSTRAINT_NAME"
            )
        }

        # Foreign key information
        if record.get("FK_CONSTRAINT_NAME"):

            column["foreign_key"] = {

                "constraint_name": record["FK_CONSTRAINT_NAME"],

                "referenced_owner": record.get(
                    "REFERENCED_OWNER"
                ),

                "referenced_table": record.get(
                    "REFERENCED_TABLE"
                ),

                "referenced_column": record.get(
                    "REFERENCED_COLUMN"
                )
            }

        tables[table_key]["columns"].append(column)

    # Final JSON structure
    schema = {

        "database": "Oracle",

        "schema_count": len({
            table["owner"]
            for table in tables.values()
        }),

        "table_count": len(tables),

        "tables": list(tables.values())
    }

    return schema


# ==================================================
# 9. SAVE JSON FILE
# ==================================================

def save_schema_json(schema):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            schema,
            file,
            indent=4,
            ensure_ascii=False,
            default=json_serializer
        )

    print(
        f"\nSchema JSON saved successfully: {OUTPUT_FILE}"
    )


# ==================================================
# 10. MAIN EXECUTION
# ==================================================

def main():

    try:

        validate_configuration()

        # Initialize Oracle client
        initialize_oracle_client()

        # Step 1: Fetch complete metadata
        metadata = fetch_schema_metadata()

        # Step 2: Remove completely empty columns
        filtered_metadata = filter_empty_columns(metadata)

        # Step 3: Convert filtered metadata to JSON
        schema = build_schema_json(filtered_metadata)

        # Step 4: Save optimized schema
        save_schema_json(schema)

        print("\n====================================")
        print("SCHEMA EXTRACTION COMPLETED")
        print("====================================")

        print(
            f"Total tables: {schema['table_count']}"
        )

        print(
            f"Total schemas: {schema['schema_count']}"
        )

        print(
            f"Total remaining columns: "
            f"{len(filtered_metadata)}"
        )

    except Exception as error:

        print(f"\nSchema extraction failed: {error}")


if __name__ == "__main__":
    main()


#after runnign this file, we will get a file named as schema.json which will have the schema of the database in json format. This file will be used in the next step to create metadata for text2sql model.