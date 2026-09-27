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


def enrich_transactions(df):
    df = df.copy()
    df['description_normalized'] = df['description_original'].apply(normalize_text)
    df['merchant_normalized'] = df['description_normalized'].apply(infer_merchant)
    df['amount_abs'] = pd.to_numeric(df['amount_abs'], errors='coerce')
    df['amount_bucket'] = df['amount_abs'].apply(amount_bucket)
    df['is_weekend'] = df['booking_weekday'].isin(['Saturday', 'Sunday'])
    df['is_income'] = df['direction'].eq('credit')
    df['is_expense'] = df['direction'].eq('debit')

    categories = df.apply(
        lambda row: infer_category(row['direction'], row['description_normalized']),
        axis=1,
        result_type='expand',
    )
    categories.columns = ['category_general', 'label_source', 'label_confidence']
    df = pd.concat([df, categories], axis=1)

    merchant_stats = (
        df.groupby(['user_id', 'merchant_normalized'], as_index=False)
        .agg(
            merchant_tx_count=('transaction_id', 'size'),
            merchant_active_months=('booking_month', 'nunique'),
            merchant_total_amount=('amount_abs', 'sum'),
        )
    )
    merchant_stats['merchant_recurring_candidate'] = (
        (merchant_stats['merchant_tx_count'] >= 3) &
        (merchant_stats['merchant_active_months'] >= 2)
    )

    df = df.merge(
        merchant_stats,
        on=['user_id', 'merchant_normalized'],
        how='left',
    )

    ordered_columns = [
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
    return df[ordered_columns]


def save_dataset(df, raw_output, clean_output):
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    clean_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_output, index=False)
    df.to_csv(clean_output, index=False)


def main():
    args = parse_args()

    print('=' * 60)
    print('ENRIQUECIMIENTO DE TRANSACCIONES TINK')
    print('=' * 60)

    df = pd.read_csv(args.input)
    enriched = enrich_transactions(df)
    save_dataset(enriched, args.raw_output, args.clean_output)

    print(f'\nTransacciones: {len(enriched):,}')
    print(f'Merchants normalizados: {enriched["merchant_normalized"].nunique():,}')
    print(f'Categorías generales: {enriched["category_general"].nunique():,}')
    print(f'Archivo raw: {args.raw_output}')
    print(f'Archivo clean: {args.clean_output}')


if __name__ == '__main__':
    main()