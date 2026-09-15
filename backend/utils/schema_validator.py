# Path: backend/utils/schema_validator.py
import json
from pathlib import Path
from typing import Any, Dict

from fastavro import validate, parse_schema


_SCHEMA_CACHE: Dict[str, Dict[str, Any]] = {}


def load_avro_schema(path: str) -> Dict[str, Any]:
    """
    Charge et parse un schéma Avro, avec cache simple en mémoire.
    """
    if path in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[path]
    
    schema_path = Path(path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schéma Avro introuvable: {path}")
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_dict = json.load(f)
    
    parsed = parse_schema(schema_dict)
    _SCHEMA_CACHE[path] = parsed
    return parsed


def validate_avro(payload: dict, schema: Dict[str, Any]) -> bool:
    """
    Valide un document Python contre un schéma Avro.
    """
    try:
        return validate(payload, schema)
    except Exception as exc:  # pragma: no cover - trace utile en debug
        print(f"[AVRO] Validation échouée: {exc}")
        return False

