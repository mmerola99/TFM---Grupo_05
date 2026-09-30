import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / 'data' / 'raw'
CLEAN_DIR = ROOT_DIR / 'data' / 'clean'
DEFAULT_INPUT = RAW_DIR / 'transacciones_tink.csv'
DEFAULT_RAW_OUTPUT = RAW_DIR / 'transacciones_tink_enriched.csv'
DEFAULT_CLEAN_OUTPUT = CLEAN_DIR / 'transacciones_tink_enriched.csv'

GENERAL_CATEGORIES = [
    'income_salary',
    'income_interest',
    'income_transfer',
    'housing',
    'groceries',
    'dining',
    'transport',
    'health',
    'education',
    'entertainment',
    'shopping',
    'subscriptions',
    'insurance',
    'taxes_fees',
    'savings_transfer',
    'cash_withdrawal',
    'debt_credit',
    'internal_transfer',
    'uncategorized',
]

MERCHANT_ALIASES = {
    'mcdonalds': [r'mc\s?donald', r'mcdonalds?'],
    'aldi': [r'\baldi\b'],
    'starbucks': [r'starbucks'],
    'coffee fellows': [r'coffee fellows'],
    'paypal': [r'paypal'],
    'carrefour': [r'carrefour'],
    'allianz': [r'allianz'],
    'netflix': [r'netflix'],
    'spotify': [r'spotify'],
    'atm': [r'\batm\b', r'cajero'],
    'rent': [r'\bmiete\b', r'alquiler', r'hipoteca', r'\brent\b', r'mortgage'],
    'salary': [r'\bsalary\b', r'nomina', r'payroll', r'haberes'],
    'social security': [r'seg social', r'seguridad social', r'social security'],
    'interest': [r'zinsertrage', r'interes', r'interest'],
    'orange loan': [r'orange loan', r'loan'],
    'irish pub': [r'irish pub', r"o'sullivan"],
}

RULES = [
    ('income_salary', 'credit', [r'\bsalary\b', r'payroll', r'nomina', r'haberes']),
    ('income_interest', 'credit', [r'zinsertrage', r'interest', r'interes']),
    ('income_transfer', 'credit', [r'seg social', r'social security', r'manutencion', r'transfer']),
    ('housing', 'debit', [r'\bmiete\b', r'alquiler', r'hipoteca', r'rent', r'mortgage']),
    ('groceries', 'debit', [r'\baldi\b', r'lidl', r'mercadona', r'carrefour', r'supermerc', r'grocery']),
    ('dining', 'debit', [r'mcdonald', r'starbucks', r'coffee fellows', r'pub', r'bar', r'restaurant', r'burger']),
    ('transport', 'debit', [r'uber', r'cabify', r'renfe', r'metro', r'bus', r'train', r'fuel', r'gas']),
    ('health', 'debit', [r'fielmann', r'pharmacy', r'farmacia', r'clinic', r'dental', r'doctor']),
    ('education', 'debit', [r'udemy', r'coursera', r'school', r'univers', r'college', r'course']),
    ('subscriptions', 'debit', [r'netflix', r'spotify', r'prime', r'disney', r'hbo']),
    ('insurance', 'debit', [r'allianz', r'insurance', r'seguro']),
    ('cash_withdrawal', 'debit', [r'\batm\b', r'cajero']),
    ('debt_credit', 'credit', [r'loan', r'credit']),
    ('shopping', 'debit', [r'paypal', r'amazon', r'zalando', r'ikea', r'store']),
    ('entertainment', 'debit', [r'cinema', r'cine', r'concert', r'festival', r'gaming', r'steam']),
]

OUTPUT_COLUMNS = [
    'user_label', 'user_id', 'profile_source', 'language_hint',
    'transaction_id', 'provider_transaction_id', 'account_id', 'account_name',
    'account_type', 'account_currency', 'account_booked_balance',
    'account_available_balance', 'account_iban', 'financial_institution_id',
    'customer_segment', 'account_last_refreshed_at', 'booked_date',
    'booking_year', 'booking_month', 'booking_day_of_month', 'booking_weekday',
    'is_weekend', 'amount_signed', 'amount_abs', 'amount_bucket',
    'amount_unscaled_value', 'amount_scale', 'currency_code', 'direction',
    'description_original', 'description_display', 'description_normalized',
    'merchant_normalized', 'transaction_type', 'status', 'provider_mutability',
    'is_income', 'is_expense', 'category_general', 'label_source',
    'label_confidence', 'merchant_tx_count', 'merchant_active_months',
    'merchant_total_amount', 'merchant_recurring_candidate',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Enriquece transacciones Tink con merchant normalizado y categoría general.'
    )
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help='CSV de transacciones canónicas de la capa 1.')
    parser.add_argument('--raw-output', type=Path, default=DEFAULT_RAW_OUTPUT,
                        help='CSV enriquecido de salida en data/raw/.')
    parser.add_argument('--clean-output', type=Path, default=DEFAULT_CLEAN_OUTPUT,
                        help='CSV enriquecido de salida en data/clean/.')
    parser.add_argument('--chunk-size', type=int, default=300000,
                        help='Filas por chunk al procesar el CSV en streaming.')
    parser.add_argument('--progress-every', type=int, default=5,
                        help='Cada cuantos chunks mostrar progreso.')
    return parser.parse_args()


def normalize_text(value):
    if pd.isna(value):
        return ''
    text = str(value).strip().lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def amount_bucket(amount_abs):
    if amount_abs < 10:
        return 'micro'
    if amount_abs < 50:
        return 'small'
    if amount_abs < 200:
        return 'medium'
    if amount_abs < 1000:
        return 'large'
    return 'xlarge'


def infer_merchant(text_normalized):
    for canonical_name, patterns in MERCHANT_ALIASES.items():
        if any(re.search(pattern, text_normalized) for pattern in patterns):
            return canonical_name

    if not text_normalized:
        return 'unknown'

    tokens = text_normalized.split()
    if len(tokens) >= 3:
        return ' '.join(tokens[:3])
    return ' '.join(tokens)


def infer_category(direction, text_normalized):
    for category, rule_direction, patterns in RULES:
        if direction != rule_direction:
            continue
        if any(re.search(pattern, text_normalized) for pattern in patterns):
            return category, 'rule', 0.9
    return 'uncategorized', 'fallback', 0.1


def add_base_enrichment(df):
    df = df.copy()
    df['description_normalized'] = df['description_original'].apply(normalize_text)
    df['merchant_normalized'] = df['description_normalized'].apply(infer_merchant)
    df['amount_abs'] = pd.to_numeric(df['amount_abs'], errors='coerce')
    df['amount_bucket'] = df['amount_abs'].apply(amount_bucket)
    df['is_weekend'] = df['booking_weekday'].isin(['Saturday', 'Sunday'])
    df['is_income'] = df['direction'].eq('credit')
    df['is_expense'] = df['direction'].eq('debit')

    category_results = [
        infer_category(direction, text)
        for direction, text in zip(df['direction'], df['description_normalized'])
    ]
    categories = pd.DataFrame(
        category_results,
        columns=['category_general', 'label_source', 'label_confidence'],
        index=df.index,
    )
    return pd.concat([df, categories], axis=1)


def accumulate_merchant_stats(df, stats):
    """Primera pasada: acumula, por (user_id, merchant_normalized), el
    numero de transacciones, los meses activos y el importe total, sin
    mantener el DataFrame completo en memoria."""
    grouped = df.groupby(['user_id', 'merchant_normalized'])
    for (user_id, merchant), group in grouped:
        key = (user_id, merchant)
        entry = stats.setdefault(key, {'count': 0, 'months': set(), 'total': 0.0})
        entry['count'] += len(group)
        entry['months'].update(group['booking_month'].dropna().unique().tolist())
        entry['total'] += group['amount_abs'].sum()


def build_and_write(input_path, raw_output, clean_output, chunk_size, progress_every):
    raw_output.parent.mkdir(parents=True, exist_ok=True)

    # --- Pasada 1: estadisticas de merchant por usuario, en streaming ---
    merchant_stats = {}
    for chunk_index, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size), start=1):
        enriched_chunk = add_base_enrichment(chunk)
        accumulate_merchant_stats(enriched_chunk, merchant_stats)
        if chunk_index % progress_every == 0:
            print(f'Pasada 1/2 (estadisticas de merchant): {chunk_index} chunks...', flush=True)

    # --- Pasada 2: recalcula enriquecimiento, añade stats y escribe ---
    total_rows = 0
    merchants_seen = set()
    categories_seen = set()
    write_header = True

    for chunk_index, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size), start=1):
        enriched_chunk = add_base_enrichment(chunk)

        counts = []
        months = []
        totals = []
        for user_id, merchant in zip(enriched_chunk['user_id'], enriched_chunk['merchant_normalized']):
            entry = merchant_stats.get((user_id, merchant))
            if entry is None:
                counts.append(0)
                months.append(0)
                totals.append(0.0)
            else:
                counts.append(entry['count'])
                months.append(len(entry['months']))
                totals.append(entry['total'])

        enriched_chunk['merchant_tx_count'] = counts
        enriched_chunk['merchant_active_months'] = months
        enriched_chunk['merchant_total_amount'] = totals
        enriched_chunk['merchant_recurring_candidate'] = (
            (enriched_chunk['merchant_tx_count'] >= 3) &
            (enriched_chunk['merchant_active_months'] >= 2)
        )

        output_chunk = enriched_chunk[OUTPUT_COLUMNS]
        output_chunk.to_csv(raw_output, index=False, mode='w' if write_header else 'a', header=write_header)
        write_header = False

        total_rows += len(output_chunk)
        merchants_seen.update(output_chunk['merchant_normalized'].unique().tolist())
        categories_seen.update(output_chunk['category_general'].unique().tolist())

        if chunk_index % progress_every == 0:
            print(f'Pasada 2/2 (escritura): {chunk_index} chunks...', flush=True)

    clean_output.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(raw_output, clean_output)

    return {
        'rows': total_rows,
        'merchants': len(merchants_seen),
        'categories': len(categories_seen),
    }


def main():
    args = parse_args()

    print('=' * 60)
    print('ENRIQUECIMIENTO DE TRANSACCIONES TINK')
    print('=' * 60)

    stats = build_and_write(
        args.input, args.raw_output, args.clean_output,
        args.chunk_size, args.progress_every,
    )

    print(f'\nTransacciones: {stats["rows"]:,}')
    print(f'Merchants normalizados: {stats["merchants"]:,}')
    print(f'Categorías generales: {stats["categories"]:,}')
    print(f'Archivo raw: {args.raw_output}')
    print(f'Archivo clean: {args.clean_output}')


if __name__ == '__main__':
    main()
