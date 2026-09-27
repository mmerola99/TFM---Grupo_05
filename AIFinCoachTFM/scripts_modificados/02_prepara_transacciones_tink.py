import argparse
import json
from pathlib import Path
import re

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / 'data' / 'raw'
CLEAN_DIR = ROOT_DIR / 'data' / 'clean'
DEFAULT_MANIFEST = ROOT_DIR / 'accounts' / 'synthetic' / 'manifest.json'
DEFAULT_RAW_OUTPUT = RAW_DIR / 'transacciones_tink.csv'
DEFAULT_CLEAN_OUTPUT = CLEAN_DIR / 'transacciones_tink.csv'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extrae una tabla transaccional canónica a partir de payloads Tink.'
    )
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST,
                        help='Ruta al manifest generado por genera_payloads_tink_sinteticos.py.')
    parser.add_argument('--raw-output', type=Path, default=DEFAULT_RAW_OUTPUT,
                        help='CSV de salida en data/raw/.')
    parser.add_argument('--clean-output', type=Path, default=DEFAULT_CLEAN_OUTPUT,
                        help='CSV de salida en data/clean/.')
    return parser.parse_args()


def load_json(path):
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def amount_from_payload(amount_payload):
    return int(amount_payload['value']['unscaledValue']) / (10 ** int(amount_payload['value']['scale']))


def numeric_user_id(user_label):
    match = re.search(r'(\d+)$', user_label)
    if not match:
        raise ValueError(f'No se pudo extraer user_id numérico de {user_label}')
    return int(match.group(1))


def account_rows_by_id(user_meta):
    accounts_path = ROOT_DIR / Path(user_meta['accounts_file'])
    accounts_payload = load_json(accounts_path)

    account_map = {}
    for account in accounts_payload.get('accounts', []):
        booked_amount = account.get('balances', {}).get('booked', {}).get('amount', {})
        available_amount = account.get('balances', {}).get('available', {}).get('amount', booked_amount)

        account_map[account['id']] = {
            'account_id': account.get('id', ''),
            'account_name': account.get('name', ''),
            'account_type': account.get('type', ''),
            'account_currency': booked_amount.get('currencyCode', ''),
            'account_booked_balance': amount_from_payload(booked_amount) if booked_amount else None,
            'account_available_balance': amount_from_payload(available_amount) if available_amount else None,
            'account_iban': account.get('identifiers', {}).get('iban', {}).get('iban', ''),
            'financial_institution_id': account.get('financialInstitutionId', ''),
            'customer_segment': account.get('customerSegment', ''),
            'account_last_refreshed_at': account.get('dates', {}).get('lastRefreshed', ''),
        }

    return account_map


def build_transaction_dataset(manifest_path):
    manifest = load_json(manifest_path)
    rows = []

    for user_meta in manifest.get('users', []):
        user_label = user_meta.get('synthetic_user', '')
        user_id = numeric_user_id(user_label)
        accounts_by_id = account_rows_by_id(user_meta)

        for relative_tx_path in user_meta.get('transaction_files', []):
            tx_path = ROOT_DIR / Path(relative_tx_path)
            tx_payload = load_json(tx_path)

            for transaction in tx_payload.get('transactions', []):
                amount_payload = transaction.get('amount', {})
                amount_value = amount_payload.get('value', {})
                account_data = accounts_by_id.get(transaction.get('accountId', ''), {})
                booked_date = transaction.get('dates', {}).get('booked', '')
                booked_ts = pd.to_datetime(booked_date, errors='coerce')
                signed_amount = amount_from_payload(amount_payload)

                rows.append({
                    'user_label': user_label,
                    'user_id': user_id,
                    'profile_source': user_meta.get('profile_source', ''),
                    'language_hint': user_meta.get('language_hint', ''),
                    'transaction_id': transaction.get('id', ''),
                    'provider_transaction_id': transaction.get('identifiers', {}).get('providerTransactionId', ''),
                    'account_id': transaction.get('accountId', ''),
                    'account_name': account_data.get('account_name', ''),
                    'account_type': account_data.get('account_type', ''),
                    'account_currency': account_data.get('account_currency', ''),
                    'account_booked_balance': account_data.get('account_booked_balance'),
                    'account_available_balance': account_data.get('account_available_balance'),
                    'account_iban': account_data.get('account_iban', ''),
                    'financial_institution_id': account_data.get('financial_institution_id', ''),
                    'customer_segment': account_data.get('customer_segment', ''),
                    'account_last_refreshed_at': account_data.get('account_last_refreshed_at', ''),
                    'booked_date': booked_date,
                    'booking_year': int(booked_ts.year) if not pd.isna(booked_ts) else None,
                    'booking_month': booked_ts.strftime('%Y-%m') if not pd.isna(booked_ts) else '',
                    'booking_day_of_month': int(booked_ts.day) if not pd.isna(booked_ts) else None,
                    'booking_weekday': booked_ts.day_name() if not pd.isna(booked_ts) else '',
                    'amount_signed': signed_amount,
                    'amount_abs': abs(signed_amount),
                    'amount_unscaled_value': amount_value.get('unscaledValue', ''),
                    'amount_scale': int(amount_value['scale']) if amount_value.get('scale') is not None else None,
                    'currency_code': amount_payload.get('currencyCode', ''),
                    'direction': 'credit' if signed_amount >= 0 else 'debit',
                    'description_original': transaction.get('descriptions', {}).get('original', ''),
                    'description_display': transaction.get('descriptions', {}).get('display', ''),
                    'transaction_type': transaction.get('types', {}).get('type', ''),
                    'status': transaction.get('status', ''),
                    'provider_mutability': transaction.get('providerMutability', ''),
                })

    if not rows:
        raise RuntimeError('No se encontraron transacciones en el manifest indicado.')

    df = pd.DataFrame(rows)
    numeric_columns = [
        'account_booked_balance',
        'account_available_balance',
        'amount_signed',
        'amount_abs',
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').round(2)
    df = df.sort_values(['user_id', 'booked_date', 'transaction_id'], ascending=[True, False, True]).reset_index(drop=True)
    return df


def save_dataset(df, raw_output, clean_output):
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    clean_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_output, index=False)
    df.to_csv(clean_output, index=False)


def main():
    args = parse_args()

    print('=' * 60)
    print('EXTRACCIÓN CANÓNICA DE TRANSACCIONES TINK')
    print('=' * 60)

    df = build_transaction_dataset(args.manifest)
    save_dataset(df, args.raw_output, args.clean_output)

    print(f'\nUsuarios: {df["user_id"].nunique():,}')
    print(f'Transacciones: {len(df):,}')
    print(f'Periodo: {df["booked_date"].min()} — {df["booked_date"].max()}')
    print(f'Cuentas únicas: {df["account_id"].nunique():,}')
    print(f'Archivo raw: {args.raw_output}')
    print(f'Archivo clean: {args.clean_output}')


if __name__ == '__main__':
    main()