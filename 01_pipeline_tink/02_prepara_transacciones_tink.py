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

OUTPUT_COLUMNS = [
    'user_label', 'user_id', 'profile_source', 'language_hint',
    'transaction_id', 'provider_transaction_id', 'account_id', 'account_name',
    'account_type', 'account_currency', 'account_booked_balance',
    'account_available_balance', 'account_iban', 'financial_institution_id',
    'customer_segment', 'account_last_refreshed_at', 'booked_date',
    'booking_year', 'booking_month', 'booking_day_of_month', 'booking_weekday',
    'amount_signed', 'amount_abs', 'amount_unscaled_value', 'amount_scale',
    'currency_code', 'direction', 'description_original', 'description_display',
    'transaction_type', 'status', 'provider_mutability',
]


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
    parser.add_argument('--users-per-batch', type=int, default=1000,
                        help='Cuantos usuarios procesar por lote antes de escribir a disco.')
    parser.add_argument('--progress-every', type=int, default=5000,
                        help='Cada cuantos usuarios mostrar progreso.')
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


def rows_for_user(user_meta):
    user_label = user_meta.get('synthetic_user', '')
    user_id = numeric_user_id(user_label)
    accounts_by_id = account_rows_by_id(user_meta)
    rows = []

    for relative_tx_path in user_meta.get('transaction_files', []):
        tx_path = ROOT_DIR / Path(relative_tx_path)
        tx_payload = load_json(tx_path)

        for transaction in tx_payload.get('transactions', []):
            amount_payload = transaction.get('amount', {})
            amount_value = amount_payload.get('value', {})
            account_data = accounts_by_id.get(transaction.get('accountId', ''), {})
            booked_date = transaction.get('dates', {}).get('booked', '')
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
    return rows


def finalize_batch(rows):
    df = pd.DataFrame(rows)
    booked_ts = pd.to_datetime(df['booked_date'], errors='coerce')
    df['booking_year'] = booked_ts.dt.year.astype('Int64')
    df['booking_month'] = booked_ts.dt.strftime('%Y-%m').fillna('')
    df['booking_day_of_month'] = booked_ts.dt.day.astype('Int64')
    df['booking_weekday'] = booked_ts.dt.day_name().fillna('')

    numeric_columns = [
        'account_booked_balance',
        'account_available_balance',
        'amount_signed',
        'amount_abs',
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').round(2)
    df = df.sort_values(['user_id', 'booked_date', 'transaction_id'], ascending=[True, False, True]).reset_index(drop=True)
    return df[OUTPUT_COLUMNS]


def build_and_write(manifest_path, raw_output, clean_output, users_per_batch, progress_every):
    manifest = load_json(manifest_path)
    users = manifest.get('users', [])

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    if raw_output.exists():
        raw_output.unlink()

    total_users = 0
    total_transactions = 0
    min_date = None
    max_date = None
    unique_accounts = set()
    write_header = True
    batch_rows = []

    def flush(batch_rows):
        nonlocal write_header, total_transactions, min_date, max_date
        if not batch_rows:
            return
        df_batch = finalize_batch(batch_rows)
        df_batch.to_csv(raw_output, index=False, mode='w' if write_header else 'a', header=write_header)
        write_header = False
        total_transactions += len(df_batch)
        unique_accounts.update(df_batch['account_id'].unique().tolist())
        batch_min = df_batch['booked_date'].min()
        batch_max = df_batch['booked_date'].max()
        nonlocal_update_dates(batch_min, batch_max)

    def nonlocal_update_dates(batch_min, batch_max):
        nonlocal min_date, max_date
        if min_date is None or (batch_min and batch_min < min_date):
            min_date = batch_min
        if max_date is None or (batch_max and batch_max > max_date):
            max_date = batch_max

    for user_meta in users:
        batch_rows.extend(rows_for_user(user_meta))
        total_users += 1

        if total_users % users_per_batch == 0:
            flush(batch_rows)
            batch_rows = []

        if total_users % progress_every == 0:
            print(f'Procesados {total_users:,}/{len(users):,} usuarios...', flush=True)

    flush(batch_rows)

    if total_transactions == 0:
        raise RuntimeError('No se encontraron transacciones en el manifest indicado.')

    clean_output.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(raw_output, clean_output)

    return {
        'users': total_users,
        'transactions': total_transactions,
        'accounts': len(unique_accounts),
        'min_date': min_date,
        'max_date': max_date,
    }


def main():
    args = parse_args()

    print('=' * 60)
    print('EXTRACCIÓN CANÓNICA DE TRANSACCIONES TINK')
    print('=' * 60)

    stats = build_and_write(
        args.manifest, args.raw_output, args.clean_output,
        args.users_per_batch, args.progress_every,
    )

    print(f'\nUsuarios: {stats["users"]:,}')
    print(f'Transacciones: {stats["transactions"]:,}')
    print(f'Periodo: {stats["min_date"]} — {stats["max_date"]}')
    print(f'Cuentas únicas: {stats["accounts"]:,}')
    print(f'Archivo raw: {args.raw_output}')
    print(f'Archivo clean: {args.clean_output}')


if __name__ == '__main__':
    main()
