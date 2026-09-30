import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / 'data' / 'raw'
CLEAN_DIR = ROOT_DIR / 'data' / 'clean'
DEFAULT_MANIFEST = ROOT_DIR / 'accounts' / 'synthetic' / 'manifest.json'
DEFAULT_IPC = RAW_DIR / 'eurostat_ipc.csv'
DEFAULT_RAW_OUTPUT = RAW_DIR / 'dataset_sintetico_usuarios_tink.csv'
DEFAULT_CLEAN_OUTPUT = CLEAN_DIR / 'dataset_final_usuarios_tink.csv'

PROFILE_ENCODING = {
    'junior': 0,
    'medio': 1,
    'senior': 2,
    'freelance': 3,
}

CATEGORY_PATTERNS = {
    'vivienda': [
        r'alquiler', r'hipoteca', r'miete', r'rent', r'landlord', r'mortgage',
        r'haus', r'wohnung',
    ],
    'alimentacion': [
        r'aldi', r'lidl', r'mercadona', r'carrefour', r'mcdonald', r'burger',
        r'restaurant', r'cafe', r'coffee', r'pub', r'bar', r'glovo',
        r'just.?eat', r'ubereats', r'deliveroo', r'super',
    ],
    'transporte': [
        r'uber', r'cabify', r'renfe', r'metro', r'bus', r'train', r'fuel',
        r'gas', r'shell', r'repsol', r'bp\b', r'parking', r'peaje',
        r'taxi', r'transit',
    ],
    'ocio': [
        r'netflix', r'spotify', r'cinema', r'cine', r'steam', r'game',
        r'ticket', r'concert', r'festival', r'gym', r'fitness',
    ],
    'salud': [
        r'farmacia', r'pharmacy', r'clinic', r'hospital', r'dental',
        r'doctor', r'salud',
    ],
    'educacion': [
        r'udemy', r'coursera', r'universidad', r'universit', r'school',
        r'college', r'edu', r'course',
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Transforma payloads Tink sinteticos en un dataset mensual por usuario.'
    )
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST,
                        help='Ruta al manifest generado por genera_payloads_tink_sinteticos.py.')
    parser.add_argument('--ipc', type=Path, default=DEFAULT_IPC,
                        help='Ruta al CSV mensual de IPC.')
    parser.add_argument('--raw-output', type=Path, default=DEFAULT_RAW_OUTPUT,
                        help='CSV mensual de salida para consumo analitico.')
    parser.add_argument('--clean-output', type=Path, default=DEFAULT_CLEAN_OUTPUT,
                        help='Copia limpia de salida en data/clean/.')
    parser.add_argument('--users-per-batch', type=int, default=2000,
                        help='Cuantos usuarios procesar por lote antes de agregar y liberar memoria.')
    parser.add_argument('--progress-every', type=int, default=8000,
                        help='Cada cuantos usuarios mostrar progreso.')
    return parser.parse_args()


def load_json(path):
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def amount_from_payload(amount_payload):
    value = amount_payload['value']
    return int(value['unscaledValue']) / (10 ** int(value['scale']))


def numeric_user_id(user_label):
    match = re.search(r'(\d+)$', user_label)
    if not match:
        raise ValueError(f'No se pudo extraer user_id numérico de {user_label}')
    return int(match.group(1))


def stable_age(user_label):
    digest = hashlib.md5(user_label.encode('utf-8')).hexdigest()
    return 25 + (int(digest[:2], 16) % 16)


def classify_category(description):
    normalized = description.lower()
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            return category
    return 'otros'


def infer_profile(monthly_income_series):
    positive_income = monthly_income_series[monthly_income_series > 0]
    if positive_income.empty:
        return 'junior'

    mean_income = positive_income.mean()
    std_income = positive_income.std(ddof=0)
    volatility_ratio = (std_income / mean_income) if mean_income else 0

    if volatility_ratio >= 0.35:
        return 'freelance'
    if mean_income < 1800:
        return 'junior'
    if mean_income < 2800:
        return 'medio'
    return 'senior'


def load_ipc_map(ipc_path):
    df_ipc = pd.read_csv(ipc_path)
    df_ipc['periodo'] = df_ipc['periodo'].astype(str)
    df_ipc['ipc'] = pd.to_numeric(df_ipc['ipc'], errors='coerce')
    return dict(zip(df_ipc['periodo'], df_ipc['ipc']))


def build_completed_ipc_map(months, ipc_map):
    ipc_series = pd.Series(ipc_map, dtype='float64')
    completed = ipc_series.reindex(sorted(months)).sort_index().ffill().bfill()
    return completed.to_dict()


def rows_for_user(user_meta):
    rows = []
    for relative_tx_path in user_meta.get('transaction_files', []):
        tx_path = ROOT_DIR / Path(relative_tx_path)
        payload = load_json(tx_path)
        for transaction in payload.get('transactions', []):
            amount = amount_from_payload(transaction['amount'])
            booked_date = transaction['dates']['booked']
            month = booked_date[:7]
            description = transaction.get('descriptions', {}).get('original', '').strip()
            expense_category = classify_category(description) if amount < 0 else None

            rows.append({
                'user_label': user_meta['synthetic_user'],
                'user_id': numeric_user_id(user_meta['synthetic_user']),
                'fecha': month,
                'amount_signed': amount,
                'descripcion': description,
                'categoria_gasto': expense_category,
                'status': transaction.get('status', 'BOOKED'),
                'profile_source': user_meta.get('profile_source', ''),
                'language_hint': user_meta.get('language_hint', ''),
            })
    return rows


def aggregate_monthly(df_tx):
    """Agrega un bloque de transacciones (de un lote de usuarios completos)
    a nivel usuario+mes. El resultado agregado es mucho mas pequeño que las
    transacciones de origen, asi que puede acumularse en memoria lote a
    lote sin problema; lo que no cabia en memoria era la lista de
    transacciones en bruto para los 40k usuarios a la vez."""
    df_tx = df_tx[df_tx['status'].eq('BOOKED')].copy()
    if df_tx.empty:
        return None

    monthly_base = (
        df_tx.groupby(['user_label', 'user_id', 'fecha', 'profile_source', 'language_hint'], as_index=False)
        .agg(
            salario=('amount_signed', lambda values: round(values[values > 0].sum(), 2)),
            gasto_total=('amount_signed', lambda values: round((-values[values < 0]).sum(), 2)),
            transacciones_mes=('amount_signed', 'size'),
        )
    )

    expense_rows = df_tx[df_tx['amount_signed'] < 0].copy()
    expense_rows['importe_gasto'] = -expense_rows['amount_signed']
    monthly_categories = (
        expense_rows.pivot_table(
            index=['user_label', 'user_id', 'fecha', 'profile_source', 'language_hint'],
            columns='categoria_gasto',
            values='importe_gasto',
            aggfunc='sum',
            fill_value=0,
        )
        .reset_index()
    )
    monthly_categories.columns.name = None

    df_monthly_batch = monthly_base.merge(
        monthly_categories,
        on=['user_label', 'user_id', 'fecha', 'profile_source', 'language_hint'],
        how='left',
    )

    for category in ['vivienda', 'alimentacion', 'transporte', 'ocio', 'salud', 'educacion', 'otros']:
        if category not in df_monthly_batch.columns:
            df_monthly_batch[category] = 0.0

    return df_monthly_batch


def build_monthly_dataset(manifest_path, ipc_map, users_per_batch, progress_every):
    manifest = load_json(manifest_path)
    users = manifest.get('users', [])

    batch_results = []
    batch_rows = []
    processed = 0

    for user_meta in users:
        batch_rows.extend(rows_for_user(user_meta))
        processed += 1

        if processed % users_per_batch == 0:
            df_tx_batch = pd.DataFrame(batch_rows)
            batch_result = aggregate_monthly(df_tx_batch)
            if batch_result is not None:
                batch_results.append(batch_result)
            batch_rows = []

        if processed % progress_every == 0:
            print(f'Procesados {processed:,}/{len(users):,} usuarios...', flush=True)

    if batch_rows:
        df_tx_batch = pd.DataFrame(batch_rows)
        batch_result = aggregate_monthly(df_tx_batch)
        if batch_result is not None:
            batch_results.append(batch_result)

    if not batch_results:
        raise RuntimeError('No se encontraron transacciones en el manifest indicado.')

    df_monthly = pd.concat(batch_results, ignore_index=True)

    df_monthly['salario'] = df_monthly['salario'].round(2)
    df_monthly['gasto_total'] = df_monthly['gasto_total'].round(2)
    df_monthly['ahorro'] = (df_monthly['salario'] - df_monthly['gasto_total']).round(2)
    df_monthly['tasa_ahorro_pct'] = df_monthly.apply(
        lambda row: round((row['ahorro'] / row['salario']) * 100, 2) if row['salario'] > 0 else 0.0,
        axis=1,
    )
    df_monthly['perfil_ahorro'] = df_monthly['tasa_ahorro_pct'].apply(classify_saving_profile)
    df_monthly['edad'] = df_monthly['user_label'].apply(stable_age)
    completed_ipc_map = build_completed_ipc_map(df_monthly['fecha'].unique(), ipc_map)
    df_monthly['ipc_mensual'] = df_monthly['fecha'].map(completed_ipc_map)
    df_monthly['ipc_mensual'] = pd.to_numeric(df_monthly['ipc_mensual'], errors='coerce')

    profile_map = (
        df_monthly.groupby('user_label')['salario']
        .apply(infer_profile)
        .to_dict()
    )
    df_monthly['perfil'] = df_monthly['user_label'].map(profile_map)
    df_monthly['perfil_encoded'] = df_monthly['perfil'].map(PROFILE_ENCODING)

    ordered_columns = [
        'user_id', 'fecha', 'edad', 'perfil', 'salario', 'vivienda', 'alimentacion',
        'transporte', 'ocio', 'salud', 'educacion', 'otros', 'gasto_total', 'ahorro',
        'tasa_ahorro_pct', 'perfil_ahorro', 'ipc_mensual', 'perfil_encoded',
        'transacciones_mes', 'profile_source', 'language_hint',
    ]

    df_monthly = df_monthly[ordered_columns].sort_values(['user_id', 'fecha']).reset_index(drop=True)
    numeric_columns = [
        'salario', 'vivienda', 'alimentacion', 'transporte', 'ocio', 'salud',
        'educacion', 'otros', 'gasto_total', 'ahorro', 'tasa_ahorro_pct',
        'ipc_mensual',
    ]
    df_monthly[numeric_columns] = df_monthly[numeric_columns].round(2)
    return df_monthly


def classify_saving_profile(saving_rate):
    if saving_rate >= 15:
        return 'buen_ahorrador'
    if saving_rate >= 5:
        return 'ahorro_moderado'
    return 'ahorro_insuficiente'


def save_dataset(df, raw_output, clean_output):
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    clean_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_output, index=False)
    df.to_csv(clean_output, index=False)


def main():
    args = parse_args()

    print('=' * 60)
    print('PREPARACIÓN DE USUARIOS DESDE PAYLOADS TINK')
    print('=' * 60)

    ipc_map = load_ipc_map(args.ipc)
    df_monthly = build_monthly_dataset(args.manifest, ipc_map, args.users_per_batch, args.progress_every)
    save_dataset(df_monthly, args.raw_output, args.clean_output)

    print(f'\nUsuarios: {df_monthly["user_id"].nunique():,}')
    print(f'Registros mensuales: {len(df_monthly):,}')
    print(f'Periodo: {df_monthly["fecha"].min()} — {df_monthly["fecha"].max()}')
    print(f'Salario medio mensual: {df_monthly["salario"].mean():.2f} €')
    print(f'Ahorro medio mensual: {df_monthly["ahorro"].mean():.2f} €')
    print(f'Archivo raw: {args.raw_output}')
    print(f'Archivo clean: {args.clean_output}')


if __name__ == '__main__':
    main()