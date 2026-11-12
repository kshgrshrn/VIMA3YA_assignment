"""Deterministic TRAIN-ONLY cleaning. Never call this on inference outputs."""
import copy
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

KEYS = ['category', 'priority', 'amount_paise', 'txn_date', 'txn_id', 'channel', 'language', 'needs_human']
CATEGORIES = {'payment_failed','refund','fraud','account_access','kyc','offers','other'}
NO_TRANSACTION = {'account_access','kyc','other'}
LEGAL = re.compile(r'\b(rbi|ombudsman|consumer court|lawyer|legal notice|legal action)\b', re.I)
REPEAT = re.compile(r'third time|second complaint|contacted support.*before|again and again|teesri baar|pehle bhi complaint|baar baar complaint', re.I)
NUMBER = r'\d[\d,]*(?:\.\d+)?'
MONEY = re.compile(r'(?:₹|\bINR\s*|\bRs\.?\s*)('+NUMBER+r')\s*(lakh|k\b)?|('+NUMBER+r')\s*(lakh\b|k\b|rupees\b|rupaye\b)', re.I)
DISTRACTOR = re.compile(r'[^.!?\n]*\b(?:balance|limit|fee|fees)\b[^!?\n]*?(?:\.(?=\s|$)|$)',re.I)


def load_rows(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')


def active_text(text):
    return '\n'.join(line for line in text.splitlines()
                     if not line.lstrip().startswith(('>', 'Received:', 'Source:')))


def amount_candidates(text):
    # Split sentences without splitting decimal numbers or the currency prefix Rs.
    sentences = re.split(r'(?<!Rs)(?<!rs)\.(?=\s|$)|[!?\n]', text)
    amounts = set()
    for sentence in sentences:
        if re.search(r'\b(balance|limit|fees?)\b|account mein abhi',sentence,re.I):
            continue
        for match in MONEY.finditer(sentence):
            number = match.group(1) or match.group(3)
            unit = (match.group(2) or match.group(4) or '').lower()
            multiplier = 100000 if unit == 'lakh' else 1000 if unit == 'k' else 1
            paise = Decimal(number.replace(',','')) * multiplier * 100
            if paise == paise.to_integral_value():
                amounts.add(int(paise))
    return amounts


def clean_rows(rows):
    """Return cleaned rows, one log entry per changed/dropped row, and audit counts.

    Semantic category/language labels are retained. We normalize representational
    errors, recompute dependent business rules, and drop uncertain amount conflicts.
    This is deliberately not a general-purpose labelling engine.
    """
    if any(not row['id'].startswith('tr-') for row in rows):
        raise ValueError('Cleaning is restricted to training IDs (tr-).')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate IDs need manual review.')
    groups = defaultdict(list)
    for row in rows:
        groups[row['messages'][1]['content']].append(row)
    conflicting = {text for text, group in groups.items()
                   if len({r['messages'][-1]['content'] for r in group}) > 1}
    clean, log, seen, findings = [], [], set(), Counter()
    for original in rows:
        row = copy.deepcopy(original)
        reasons, drop = [], False
        text = row['messages'][1]['content']
        body = active_text(text)
        if text in conflicting:
            reasons.append('conflicting_labels_for_identical_input'); drop = True
        elif text in seen:
            reasons.append('exact_duplicate_input'); drop = True
        seen.add(text)
        try:
            label = json.loads(row['messages'][-1]['content'])
        except (json.JSONDecodeError, TypeError):
            label = None
        if not isinstance(label, dict):
            reasons.append('invalid_json_label'); drop = True
        if not drop:
            if 'prio' in label and 'priority' not in label:
                old = label.pop('prio')
                mapping = {'low':'P3','medium':'P2','high':'P1'}
                label['priority'] = mapping.get(old, old)
                reasons.append('legacy_priority_key_and_value')
            if set(label) != set(KEYS):
                reasons.append('unexpected_schema_keys'); drop = True
        if not drop:
            if label['category'] not in CATEGORIES or label['language'] not in {'en','hi-en'}:
                reasons.append('invalid_category_or_language'); drop = True
            if type(label['needs_human']) is not bool:
                reasons.append('invalid_boolean_type'); drop = True
            amount = label['amount_paise']
            if amount is not None and (type(amount) is not int or amount < 0):
                reasons.append('invalid_amount_type_or_sign'); drop = True
            if label['channel'] not in {None,'upi','card','netbanking','wallet'}:
                reasons.append('invalid_channel'); drop = True
        if not drop:
            if label['category'] in NO_TRANSACTION:
                for key in ['amount_paise','txn_date','txn_id','channel']:
                    if label[key] is not None:
                        label[key] = None; reasons.append('null_transaction_fields_for_category')
            else:
                candidates = amount_candidates(body)
                if len(candidates) == 1 and label['amount_paise'] is not None:
                    extracted = next(iter(candidates))
                    if extracted != label['amount_paise']:
                        # Only a provable unit-scale error is automatically repaired.
                        if extracted // 100 == label['amount_paise']:
                            label['amount_paise'] = extracted; reasons.append('rupees_stored_as_paise')
                        else:
                            reasons.append('ambiguous_amount_mismatch_drop'); drop = True
                elif len(candidates) > 1:
                    reasons.append('multiple_claim_amount_candidates_drop'); drop = True
            date = label['txn_date']
            if date is not None:
                if not isinstance(date,str):
                    reasons.append('invalid_date_type'); drop = True
                else:
                    try:
                        if re.fullmatch(r'\d{2}/\d{2}/\d{4}',date):
                            label['txn_date'] = datetime.strptime(date,'%d/%m/%Y').date().isoformat()
                            reasons.append('legacy_date_to_iso')
                        elif not re.fullmatch(r'\d{4}-\d{2}-\d{2}',date):
                            raise ValueError('Unrecognized date')
                        datetime.strptime(label['txn_date'],'%Y-%m-%d')
                    except ValueError:
                        reasons.append('invalid_date_drop'); drop = True
            transaction_id = label['txn_id']
            if transaction_id is not None:
                normalized = re.sub(r'[^A-Z0-9]','',str(transaction_id).upper())
                if not re.fullmatch(r'NP\d{10}',normalized):
                    reasons.append('invalid_transaction_id_drop'); drop = True
                elif normalized != transaction_id:
                    label['txn_id'] = normalized; reasons.append('normalize_transaction_id')
            if not drop:
                amount = label['amount_paise'] or 0
                if label['category'] == 'fraud' or amount >= 5000000 or LEGAL.search(body):
                    priority = 'P1'
                elif label['category'] == 'account_access' or (label['category'] in {'refund','payment_failed'} and amount >= 500000):
                    priority = 'P2'
                else:
                    priority = 'P3'
                if label['priority'] != priority:
                    label['priority'] = priority; reasons.append('priority_rule_recomputed')
                # Preserve existing evidence of prior contact; only force true when
                # explicit rules establish it. Do not guess false from no regex match.
                if (priority == 'P1' or REPEAT.search(body)) and not label['needs_human']:
                    label['needs_human'] = True; reasons.append('human_escalation_rule')
                normalized_label = {key:label[key] for key in KEYS}
                # Do not log harmless whitespace/key-order differences as corrections.
                row['messages'][-1]['content'] = json.dumps(normalized_label,ensure_ascii=False)
                clean.append(row)
        if reasons:
            reasons = list(dict.fromkeys(reasons))
            findings.update(reasons)
            log.append({'id':row['id'],'action':'drop' if drop else 'fix','reason':'; '.join(reasons)})
    summary = {'raw_rows':len(rows),'clean_rows':len(clean),'dropped':sum(x['action']=='drop' for x in log),
               'fixed':sum(x['action']=='fix' for x in log),'findings':dict(findings),
               'limitations':['No automatic category/language relabelling','No complete semantic date parser',
                              'Ambiguous amount discrepancies are dropped, not guessed',
                              'Regexes for repeated contact are not exhaustive; existing true labels retained']}
    return clean, log, summary


def audit_file(train_path, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    rows, log, summary = clean_rows(load_rows(train_path))
    save_rows(output_dir/'clean_train.jsonl',rows)
    with (output_dir/'cleaning_log.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','action','reason']);writer.writeheader();writer.writerows(log)
    (output_dir/'audit_summary.json').write_text(json.dumps(summary,indent=2))
    return rows, summary
