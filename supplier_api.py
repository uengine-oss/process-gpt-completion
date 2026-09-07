import asyncio
import os
import re
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

import psycopg2
from fastapi import FastAPI, HTTPException, Query
from psycopg2.extras import RealDictCursor, execute_values

from database import db_config_var, subdomain_var


SUPPLIER_TABLE_NAME = "suppliers"
DEFAULT_SUPPLIER_XLSX_PATH = Path(__file__).resolve().parent / "20260416161128.xlsx"
DEFAULT_IMPORT_BATCH_SIZE = 500
MAX_IMPORT_BATCH_SIZE = 2_000
MAX_SEARCH_LIMIT = 500
TRUTHY_VALUES = {"1", "true", "yes", "on"}

XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XLSX_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
XLSX_NS = {"a": XLSX_MAIN_NS, "r": XLSX_REL_NS}
XLSX_RELATIONSHIP_NS = {"rel": XLSX_PACKAGE_REL_NS}

REQUIRED_HEADERS = {
    "registration_type": "등록유형",
    "registered_at": "등록및변경일자",
    "name": "공급업체명",
    "business_number": "사업자번호",
}

SUPPLIER_UUID_NAMESPACE = uuid.UUID("d7cdb3a5-c8be-43f7-a0ec-4562928218be")


class XlsxReadError(ValueError):
    pass


def add_routes_to_app(app: FastAPI):
    app.add_api_route("/suppliers/search", search_suppliers, methods=["GET"])
    app.add_api_route("/admin/suppliers/import", import_default_suppliers, methods=["POST"])


async def search_suppliers(
    keyword: Optional[str] = Query(None, description="검색할 공급업체명"),
    q: Optional[str] = Query(None, description="keyword와 동일한 검색어 alias"),
    page: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=MAX_SEARCH_LIMIT),
):
    resolved_keyword = ((keyword if keyword is not None else q) or "").strip()
    if not resolved_keyword:
        raise HTTPException(status_code=400, detail="keyword is required")

    tenant_id = _get_current_tenant_id()
    result = await asyncio.to_thread(_search_suppliers, resolved_keyword, page, limit, tenant_id)
    return {
        "tenant_id": tenant_id,
        "keyword": resolved_keyword,
        "page": page,
        "limit": limit,
        **result,
    }


async def import_default_suppliers(
    batch_size: int = Query(DEFAULT_IMPORT_BATCH_SIZE, ge=1, le=MAX_IMPORT_BATCH_SIZE),
):
    _ensure_supplier_import_enabled()
    tenant_id = _get_current_tenant_id()
    summary = await asyncio.to_thread(
        _import_suppliers_from_xlsx,
        DEFAULT_SUPPLIER_XLSX_PATH,
        tenant_id,
        batch_size,
    )
    return {
        "tenant_id": tenant_id,
        "source": str(DEFAULT_SUPPLIER_XLSX_PATH.name),
        "summary": summary,
    }


def _ensure_supplier_import_enabled() -> None:
    enabled = (os.getenv("ENABLE_SUPPLIER_IMPORT_ADMIN") or "").strip().lower()
    if enabled in TRUTHY_VALUES:
        return

    environment = (os.getenv("ENV") or "").strip().lower()
    if environment != "production":
        return

    raise HTTPException(status_code=404, detail="Not Found")


def _get_current_tenant_id() -> str:
    tenant_id = (subdomain_var.get() or "").strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id is not resolved for this request")
    return tenant_id


def _connect_db():
    db_config = db_config_var.get() or {}
    required_keys = ("dbname", "user", "host", "port")
    missing_keys = [key for key in required_keys if not str(db_config.get(key) or "").strip()]
    if missing_keys:
        raise HTTPException(status_code=500, detail=f"Database configuration is missing: {', '.join(missing_keys)}")
    try:
        return psycopg2.connect(**db_config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to connect to database: {exc}") from exc


def _search_suppliers(keyword: str, page: int, limit: int, tenant_id: str) -> Dict[str, Any]:
    normalized_keyword = _normalize_supplier_name(keyword)
    business_keyword = _normalize_business_number(keyword)
    offset = page * limit

    connection = None
    cursor = None
    try:
        connection = _connect_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        _ensure_suppliers_table(cursor)
        connection.commit()

        params = {
            "tenant_id": tenant_id,
            "normalized_keyword": normalized_keyword,
            "business_keyword": business_keyword,
            "limit": limit,
            "offset": offset,
        }
        where_clause = """
            tenant_id = %(tenant_id)s
            AND (
                strpos(normalized_name, %(normalized_keyword)s) > 0
                OR (
                    %(business_keyword)s <> ''
                    AND strpos(coalesce(business_number, ''), %(business_keyword)s) > 0
                )
            )
        """

        cursor.execute(
            f"SELECT count(*) AS total FROM {SUPPLIER_TABLE_NAME} WHERE {where_clause}",
            params,
        )
        total = int(cursor.fetchone()["total"])

        cursor.execute(
            f"""
            SELECT
                id::text,
                name,
                registration_type,
                registered_at,
                business_number,
                source_file,
                source_sheet,
                source_row,
                updated_at
            FROM {SUPPLIER_TABLE_NAME}
            WHERE {where_clause}
            ORDER BY
                CASE
                    WHEN normalized_name = %(normalized_keyword)s THEN 0
                    WHEN left(normalized_name, length(%(normalized_keyword)s)) = %(normalized_keyword)s THEN 1
                    ELSE 2
                END,
                name ASC,
                business_number ASC NULLS LAST
            LIMIT %(limit)s OFFSET %(offset)s
            """,
            params,
        )
        rows = [_serialize_supplier_row(row) for row in cursor.fetchall()]

        return {
            "total": total,
            "items": rows,
        }
    except HTTPException:
        if connection is not None:
            connection.rollback()
        raise
    except Exception as exc:
        if connection is not None:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to search suppliers: {exc}") from exc
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


def _import_suppliers_from_xlsx(xlsx_path: Path, tenant_id: str, batch_size: int = DEFAULT_IMPORT_BATCH_SIZE) -> Dict[str, Any]:
    supplier_rows = _extract_supplier_rows_from_xlsx(xlsx_path)
    records = _build_supplier_records(supplier_rows, tenant_id, xlsx_path.name)

    connection = None
    cursor = None
    try:
        connection = _connect_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        _ensure_suppliers_table(cursor)
        cursor.execute(
            f"DELETE FROM {SUPPLIER_TABLE_NAME} WHERE tenant_id = %s AND source_file = %s",
            (tenant_id, xlsx_path.name),
        )
        inserted_or_updated = _upsert_supplier_records(cursor, records, batch_size)
        connection.commit()
    except HTTPException:
        if connection is not None:
            connection.rollback()
        raise
    except Exception as exc:
        if connection is not None:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to import suppliers: {exc}") from exc
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

    return {
        "status_message": "REPLACED",
        "extracted_rows": len(supplier_rows),
        "unique_records": len(records),
        "inserted_or_updated": inserted_or_updated,
        "deduplicated_rows": len(supplier_rows) - len(records),
        "table_name": SUPPLIER_TABLE_NAME,
    }


def _ensure_suppliers_table(cursor) -> None:
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SUPPLIER_TABLE_NAME} (
            id uuid PRIMARY KEY,
            tenant_id text NOT NULL,
            name text NOT NULL,
            normalized_name text NOT NULL,
            registration_type text,
            registered_at date,
            business_number text,
            source_file text NOT NULL,
            source_sheet text NOT NULL,
            source_row integer NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    cursor.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {SUPPLIER_TABLE_NAME}_tenant_normalized_name_idx
            ON {SUPPLIER_TABLE_NAME} (tenant_id, normalized_name)
        """
    )
    cursor.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {SUPPLIER_TABLE_NAME}_tenant_business_number_idx
            ON {SUPPLIER_TABLE_NAME} (tenant_id, business_number)
        """
    )


def _upsert_supplier_records(cursor, records: List[Dict[str, Any]], batch_size: int) -> int:
    if not records:
        return 0

    values = [
        (
            record["id"],
            record["tenant_id"],
            record["name"],
            record["normalized_name"],
            record["registration_type"],
            record["registered_at"],
            record["business_number"],
            record["source_file"],
            record["source_sheet"],
            record["source_row"],
        )
        for record in records
    ]
    execute_values(
        cursor,
        f"""
        INSERT INTO {SUPPLIER_TABLE_NAME} (
            id,
            tenant_id,
            name,
            normalized_name,
            registration_type,
            registered_at,
            business_number,
            source_file,
            source_sheet,
            source_row
        )
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            normalized_name = EXCLUDED.normalized_name,
            registration_type = EXCLUDED.registration_type,
            registered_at = EXCLUDED.registered_at,
            business_number = EXCLUDED.business_number,
            source_file = EXCLUDED.source_file,
            source_sheet = EXCLUDED.source_sheet,
            source_row = EXCLUDED.source_row,
            updated_at = now()
        """,
        values,
        page_size=batch_size,
    )
    return len(records)


def _extract_supplier_rows_from_xlsx(xlsx_path: Path) -> List[Dict[str, Any]]:
    if not xlsx_path.exists():
        raise HTTPException(status_code=404, detail=f"XLSX file not found: {xlsx_path}")

    rows_by_sheet = _read_xlsx_rows(xlsx_path)
    supplier_rows: List[Dict[str, Any]] = []

    for sheet_name, rows in rows_by_sheet.items():
        header_index = _find_header_row_index(rows)
        if header_index is None:
            continue

        headers = {
            _normalize_header(cell_value): column_index
            for column_index, cell_value in rows[header_index].items()
            if _normalize_header(cell_value)
        }
        missing_headers = [
            header
            for header in REQUIRED_HEADERS.values()
            if _normalize_header(header) not in headers
        ]
        if missing_headers:
            raise HTTPException(
                status_code=400,
                detail=f"Required headers are missing in {sheet_name}: {', '.join(missing_headers)}",
            )

        column_map = {
            field_name: headers[_normalize_header(header)]
            for field_name, header in REQUIRED_HEADERS.items()
        }
        for row in rows[header_index + 1:]:
            name = _clean_cell_text(row.get(column_map["name"]))
            if not name:
                continue

            supplier_rows.append(
                {
                    "name": name,
                    "registration_type": _clean_cell_text(row.get(column_map["registration_type"])) or None,
                    "registered_at": _parse_excel_date(row.get(column_map["registered_at"])),
                    "business_number": _clean_cell_text(row.get(column_map["business_number"])) or None,
                    "source_sheet": sheet_name,
                    "source_row": int(row.get("_row_number", 0) or 0),
                }
            )

    if not supplier_rows:
        raise HTTPException(status_code=400, detail="No supplier rows were found in the XLSX file")

    return supplier_rows


def _read_xlsx_rows(xlsx_path: Path) -> Dict[str, List[Dict[Any, Any]]]:
    try:
        with ZipFile(xlsx_path) as archive:
            shared_strings = _read_shared_strings(archive)
            sheet_targets = _read_sheet_targets(archive)
            rows_by_sheet: Dict[str, List[Dict[Any, Any]]] = {}
            for sheet_name, target in sheet_targets:
                rows_by_sheet[sheet_name] = _read_sheet_rows(archive, target, shared_strings)
            return rows_by_sheet
    except BadZipFile as exc:
        raise HTTPException(status_code=400, detail=f"Invalid XLSX file: {xlsx_path}") from exc
    except XlsxReadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read XLSX file: {exc}") from exc


def _read_shared_strings(archive: ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    shared_strings: List[str] = []
    for shared_item in root.findall("a:si", XLSX_NS):
        text_parts = [
            text_node.text or ""
            for text_node in shared_item.iter(f"{{{XLSX_MAIN_NS}}}t")
        ]
        shared_strings.append("".join(text_parts))
    return shared_strings


def _read_sheet_targets(archive: ZipFile) -> List[tuple[str, str]]:
    try:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    except KeyError as exc:
        raise XlsxReadError("XLSX workbook metadata is missing") from exc

    relationship_targets = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in relationships.findall("rel:Relationship", XLSX_RELATIONSHIP_NS)
    }

    sheets_node = workbook.find("a:sheets", XLSX_NS)
    if sheets_node is None:
        raise XlsxReadError("XLSX workbook does not contain sheets")

    sheet_targets: List[tuple[str, str]] = []
    for sheet in sheets_node.findall("a:sheet", XLSX_NS):
        relationship_id = sheet.attrib.get(f"{{{XLSX_REL_NS}}}id")
        target = relationship_targets.get(relationship_id or "")
        if not target:
            continue
        sheet_targets.append((sheet.attrib.get("name", "Sheet"), _normalize_sheet_target(target)))

    if not sheet_targets:
        raise XlsxReadError("XLSX workbook does not contain readable sheet relationships")
    return sheet_targets


def _normalize_sheet_target(target: str) -> str:
    target = target.lstrip("/")
    if target.startswith("xl/"):
        return target
    return f"xl/{target}"


def _read_sheet_rows(archive: ZipFile, sheet_path: str, shared_strings: List[str]) -> List[Dict[Any, Any]]:
    try:
        root = ET.fromstring(archive.read(sheet_path))
    except KeyError as exc:
        raise XlsxReadError(f"Worksheet file is missing: {sheet_path}") from exc

    rows: List[Dict[Any, Any]] = []
    for row_node in root.findall(".//a:sheetData/a:row", XLSX_NS):
        row: Dict[Any, Any] = {"_row_number": int(float(row_node.attrib.get("r", "0") or 0))}
        for cell_node in row_node.findall("a:c", XLSX_NS):
            reference = cell_node.attrib.get("r", "")
            if not reference:
                continue
            row[_column_index_from_reference(reference)] = _read_cell_value(cell_node, shared_strings)
        rows.append(row)
    return rows


def _read_cell_value(cell_node: ET.Element, shared_strings: List[str]) -> Optional[str]:
    cell_type = cell_node.attrib.get("t")
    value_node = cell_node.find("a:v", XLSX_NS)

    if cell_type == "inlineStr":
        return "".join(
            text_node.text or ""
            for text_node in cell_node.iter(f"{{{XLSX_MAIN_NS}}}t")
        )

    if value_node is None or value_node.text is None:
        return None

    value = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(value)]
        except (IndexError, ValueError) as exc:
            raise XlsxReadError(f"Invalid shared string index: {value}") from exc
    return value


def _find_header_row_index(rows: List[Dict[Any, Any]]) -> Optional[int]:
    required_header = _normalize_header(REQUIRED_HEADERS["name"])
    for index, row in enumerate(rows):
        normalized_values = {_normalize_header(value) for key, value in row.items() if key != "_row_number"}
        if required_header in normalized_values:
            return index
    return None


def _build_supplier_records(supplier_rows: List[Dict[str, Any]], tenant_id: str, source_file: str) -> List[Dict[str, Any]]:
    records_by_id: Dict[str, Dict[str, Any]] = {}
    for row in supplier_rows:
        normalized_name = _normalize_supplier_name(row["name"])
        business_number = _normalize_business_number(row.get("business_number"))
        identity_key = business_number or normalized_name
        supplier_id = str(uuid.uuid5(SUPPLIER_UUID_NAMESPACE, f"{tenant_id}:{identity_key}"))
        record = {
            "id": supplier_id,
            "tenant_id": tenant_id,
            "name": row["name"],
            "normalized_name": normalized_name,
            "registration_type": row.get("registration_type"),
            "registered_at": row.get("registered_at"),
            "business_number": business_number or None,
            "source_file": source_file,
            "source_sheet": row.get("source_sheet") or "",
            "source_row": row.get("source_row") or 0,
        }

        existing_record = records_by_id.get(supplier_id)
        if existing_record is None or _record_is_newer(record, existing_record):
            records_by_id[supplier_id] = record

    return list(records_by_id.values())


def _record_is_newer(candidate: Dict[str, Any], existing: Dict[str, Any]) -> bool:
    candidate_date = candidate.get("registered_at")
    existing_date = existing.get("registered_at")
    if candidate_date and existing_date and candidate_date != existing_date:
        return candidate_date > existing_date
    if candidate_date and not existing_date:
        return True
    if not candidate_date and existing_date:
        return False
    return int(candidate.get("source_row") or 0) > int(existing.get("source_row") or 0)


def _parse_excel_date(value: Any) -> Optional[date]:
    cleaned_value = _clean_cell_text(value)
    if not cleaned_value:
        return None

    for date_format in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned_value, date_format).date()
        except ValueError:
            pass

    try:
        serial_value = float(cleaned_value)
    except ValueError:
        return None

    if serial_value <= 0:
        return None
    return date(1899, 12, 30) + timedelta(days=int(serial_value))


def _serialize_supplier_row(row: Dict[str, Any]) -> Dict[str, Any]:
    serialized = dict(row)
    for key in ("registered_at", "updated_at"):
        value = serialized.get(key)
        if hasattr(value, "isoformat"):
            serialized[key] = value.isoformat()
    return serialized


def _column_index_from_reference(reference: str) -> int:
    column_letters_match = re.match(r"([A-Z]+)", reference.upper())
    if column_letters_match is None:
        raise XlsxReadError(f"Invalid cell reference: {reference}")

    column_number = 0
    for letter in column_letters_match.group(1):
        column_number = column_number * 26 + ord(letter) - ord("A") + 1
    return column_number - 1


def _clean_cell_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_header(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def _normalize_supplier_name(value: Any) -> str:
    return re.sub(r"\s+", "", _clean_cell_text(value).lower())


def _normalize_business_number(value: Any) -> str:
    return re.sub(r"\D+", "", str(value or ""))

