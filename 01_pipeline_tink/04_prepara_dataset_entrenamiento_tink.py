import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / 'data' / 'raw'
CLEAN_DIR = ROOT_DIR / 'data' / 'clean'
DEFAULT_INPUT = RAW_DIR / 'transacciones_tink_enriched.csv'
DEFAULT_RAW_OUTPUT = RAW_DIR / 'transacciones_tink_model_input.csv'
DEFAULT_CLEAN_OUTPUT = CLEAN_DIR / 'transacciones_tink_model_input.csv'
DEFAULT_REVIEW_OUTPUT = RAW_DIR / 'transacciones_tink_label_review.csv'

REQUIRED_COLUMNS = [
    'transaction_id', 'provider_transaction_id', 'user_id', 'user_label',
    'profile_source', 'language_hint', 'booked_date', 'booking_year',
    'booking_month', 'booking_day_of_month', 'booking_weekday', 'is_weekend',
    'account_id', 'account_type', 'account_currency', 'customer_segment',
    'direction', 'amount_signed', 'amount_abs', 'amount_bucket', 'currency_code',
    'description_original', 'description_display', 'description_normalized',
    'merchant_normalized', 'category_general', 'label_source', 'label_confidence',
    'merchant_tx_count', 'merchant_active_months', 'merchant_total_amount',
    'merchant_recurring_candidate', 'is_income', 'is_expense',
]

TRAINING_COLUMNS = [
    'transaction_id', 'provider_transaction_id', 'user_id', 'user_label',
    'profile_source', 'language_hint', 'booked_date', 'booking_year',
    'booking_month', 'booking_day_of_month', 'booking_weekday', 'is_weekend',
    'account_id', 'account_type', 'account_currency', 'customer_segment',
    'direction', 'amount_signed', 'amount_abs', 'amount_bucket', 'currency_code',
    'description_original', 'description_display', 'description_normalized',
    'text_token_count', 'merchant_normalized', 'merchant_known',
    'merchant_tx_count', 'merchant_active_months', 'merchant_total_amount',
    'merchant_recurring_candidate', 'is_income', 'is_expense',
    'target_category', 'target_class_count', 'target_is_rare',
    'label_source', 'label_confidence', 'label_quality', 'dataset_split',
]

REVIEW_COLUMNS = [
    'transaction_id', 'provider_transaction_id', 'user_id', 'user_label',
    'booked_date', 'direction', 'amount_signed', 'amount_abs',
    'description_original', 'description_display', 'description_normalized',
    'merchant_normalized', 'category_general', 'label_source',
    'label_confidence', 'review_reason', 'merchant_tx_count', 'merchant_active_months',
    'merchant_recurring_candidate',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepara un dataset de entrenamiento a partir de transacciones Tink enriquecidas.'
    )
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help='CSV enriquecido de capa 2.')
    parser.add_argument('--raw-output', type=Path, default=DEFAULT_RAW_OUTPUT,
                        help='CSV entrenable de salida en data/raw/.')
    parser.add_argument('--clean-output', type=Path, default=DEFAULT_CLEAN_OUTPUT,
                        help='CSV entrenable de salida en data/clean/.')
    parser.add_argument('--review-output', type=Path, default=DEFAULT_REVIEW_OUTPUT,
                        help='CSV con transacciones a revisar manualmente.')
    parser.add_argument('--min-confidence', type=float, default=0.8,
                        help='Confianza mínima para aceptar weak labels en entrenamiento.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria para particionado reproducible.')
    parser.add_argument('--chunk-size', type=int, default=300000,
                        help='Filas por chunk al procesar el CSV en streaming.')
    parser.add_argument('--progress-every', type=int, default=5,
                        help='Cada cuantos chunks mostrar progreso.')
    return parser.parse_args()


def derive_columns(df, min_confidence):
    df = df.copy()
    df['label_confidence'] = pd.to_numeric(df['label_confidence'], errors='coerce').fillna(0.0)
    df['amount_abs'] = pd.to_numeric(df['amount_abs'], errors='coerce').fillna(0.0)
    df['amount_signed'] = pd.to_numeric(df['amount_signed'], errors='coerce').fillna(0.0)
    df['merchant_tx_count'] = pd.to_numeric(df['merchant_tx_count'], errors='coerce').fillna(0).astype(int)
    df['merchant_active_months'] = pd.to_numeric(df['merchant_active_months'], errors='coerce').fillna(0).astype(int)
    df['merchant_total_amount'] = pd.to_numeric(df['merchant_total_amount'], errors='coerce').fillna(0.0)
    df['booking_day_of_month'] = pd.to_numeric(df['booking_day_of_month'], errors='coerce').fillna(0).astype(int)
    df['merchant_normalized'] = df['merchant_normalized'].fillna('unknown')
    df['description_normalized'] = df['description_normalized'].fillna('')

    df['merchant_known'] = df['merchant_normalized'].ne('unknown')
    df['text_token_count'] = df['description_normalized'].str.split().str.len().fillna(0).astype(int)
    df['target_category'] = df['category_general']
    df['label_quality'] = df['label_source'].map({'rule': 'strong', 'fallback': 'weak'}).fillna('weak')
    df['review_reason'] = 'accepted'

    low_confidence_mask = df['label_confidence'].lt(min_confidence)
    uncategorized_mask = df['target_category'].eq('uncategorized')
    df.loc[low_confidence_mask, 'review_reason'] = 'low_confidence'
    df.loc[uncategorized_mask, 'review_reason'] = 'uncategorized'
    df.loc[low_confidence_mask & uncategorized_mask, 'review_reason'] = 'uncategorized_low_confidence'

    accepted_mask = (
        df['target_category'].ne('uncategorized') &
        df['label_confidence'].ge(min_confidence)
    )
    return df, accepted_mask


def build_user_split_map(user_ids, seed):
    """Particiona por user_id completo (todas las transacciones de un mismo
    usuario caen en un unico split). Ver nota de diseno en
    01_genera_payloads_tink_sinteticos.py: una particion por texto exacto
    de transaccion colapsaba con este dataset, porque el vocabulario de
    descripciones Tink sintetico es finito y se repite entre usuarios.
    Particionar por usuario evita ese colapso y encaja con el escenario de
    negocio (generalizar a usuarios nuevos que conectan Tink)."""
    rng = random.Random(seed)
    user_ids = list(user_ids)
    rng.shuffle(user_ids)
    size = len(user_ids)

    test_size = max(1, int(round(size * 0.15)))
    validation_size = max(1, int(round(size * 0.15)))
    while size - test_size - validation_size < 1 and validation_size > 0:
        validation_size -= 1
    while size - test_size - validation_size < 1 and test_size > 0:
        test_size -= 1

    test_users = set(user_ids[:test_size])
    validation_users = set(user_ids[test_size:test_size + validation_size])

    split_map = {}
    for user_id in user_ids:
        if user_id in test_users:
            split_map[user_id] = 'test'
        elif user_id in validation_users:
            split_map[user_id] = 'validation'
        else:
            split_map[user_id] = 'train'
    return split_map


def collect_metadata(input_path, min_confidence, chunk_size, progress_every):
    class_counter = Counter()
    user_ids = set()

    for chunk_index, chunk in enumerate(
        pd.read_csv(input_path, usecols=REQUIRED_COLUMNS, chunksize=chunk_size), start=1
    ):
        derived, accepted_mask = derive_columns(chunk, min_confidence)
        user_ids.update(derived['user_id'].unique().tolist())
        class_counter.update(derived.loc[accepted_mask, 'target_category'].tolist())
        if chunk_index % progress_every == 0:
            print(f'Pasada 1/2 (metadatos): {chunk_index} chunks...', flush=True)

    return class_counter, user_ids


def build_and_write(input_path, raw_output, review_output, min_confidence, seed, chunk_size, progress_every):
    class_counter, user_ids = collect_metadata(input_path, min_confidence, chunk_size, progress_every)
    split_map = build_user_split_map(user_ids, seed)

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    review_output.parent.mkdir(parents=True, exist_ok=True)

    training_rows = 0
    review_rows = 0
    split_counts = Counter()
    category_counts = Counter()
    write_training_header = True
    write_review_header = True

    for chunk_index, chunk in enumerate(
        pd.read_csv(input_path, usecols=REQUIRED_COLUMNS, chunksize=chunk_size), start=1
    ):
        derived, accepted_mask = derive_columns(chunk, min_confidence)

        training_chunk = derived.loc[accepted_mask].copy()
        review_chunk = derived.loc[~accepted_mask].copy()

        if not training_chunk.empty:
            training_chunk['target_class_count'] = training_chunk['target_category'].map(class_counter).astype(int)
            training_chunk['target_is_rare'] = training_chunk['target_class_count'].lt(5)
            training_chunk['dataset_split'] = training_chunk['user_id'].map(split_map)
            training_out = training_chunk[TRAINING_COLUMNS]
            training_out.to_csv(raw_output, index=False, mode='w' if write_training_header else 'a', header=write_training_header)
            write_training_header = False
            training_rows += len(training_out)
            split_counts.update(training_out['dataset_split'].tolist())
            category_counts.update(training_out['target_category'].tolist())

        if not review_chunk.empty:
            review_out = review_chunk[REVIEW_COLUMNS]
            review_out.to_csv(review_output, index=False, mode='w' if write_review_header else 'a', header=write_review_header)
            write_review_header = False
            review_rows += len(review_out)

        if chunk_index % progress_every == 0:
            print(f'Pasada 2/2 (escritura): {chunk_index} chunks...', flush=True)

    if training_rows == 0:
        raise RuntimeError('No quedaron transacciones de entrenamiento tras aplicar los filtros.')

    return {
        'training_rows': training_rows,
        'review_rows': review_rows,
        'categories': len(category_counts),
        'users': len({uid for uid, split in split_map.items()}),
        'split_counts': split_counts,
        'top_categories': category_counts.most_common(10),
    }


def save_dataset_copy(source_path, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(source_path, dest_path)


def main():
    args = parse_args()

    print('=' * 60)
    print('PREPARACION DE DATASET DE ENTRENAMIENTO TINK')
    print('=' * 60)

    stats = build_and_write(
        args.input, args.raw_output, args.review_output,
        args.min_confidence, args.seed, args.chunk_size, args.progress_every,
    )
    save_dataset_copy(args.raw_output, args.clean_output)

    print(f'\nEntrenamiento: {stats["training_rows"]:,} transacciones')
    print(f'Revision manual: {stats["review_rows"]:,} transacciones')
    print(f'Categorias objetivo: {stats["categories"]:,}')
    print(f'Usuarios en entrenamiento: {stats["users"]:,}')
    print('\nDistribucion por split:')
    for split_name, count in stats['split_counts'].items():
        print(f'  {split_name}: {count:,}')
    print('\nTop categorias:')
    for category, count in stats['top_categories']:
        print(f'  {category}: {count:,}')
    print(f'\nArchivo raw: {args.raw_output}')
    print(f'Archivo clean: {args.clean_output}')
    print(f'Archivo review: {args.review_output}')


if __name__ == '__main__':
    main()
