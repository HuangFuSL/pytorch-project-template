import csv
import json
from typing import Any, Dict, List


def dumpData(records: List[Dict[str, Any]], path: str, format: str):
    if format == 'csv':
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    elif format == 'json':
        with open(path, 'w') as f:
            json.dump(records, f)
