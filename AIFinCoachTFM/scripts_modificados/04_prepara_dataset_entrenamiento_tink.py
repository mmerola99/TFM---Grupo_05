import argparse
import random
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
    return parser.parse_args()


def assign_split(df, seed):
    rng = random.Random(seed)
    split_by_index = {}

    for category_name, category_group in df.groupby('target_category'):
        group_keys = list(category_group['split_group'].drop_duplicates())
        rng.shuffle(group_keys)
        size = len(group_keys)

        if size == 1:
            for group_key in group_keys:
                mask = category_group['split_group'].eq(group_key)
                for index in category_group.loc[mask].index:
                    split_by_index[index] = 'train'
            continue

        if size == 2:
            evaluation_split = 'validation' if (sum(ord(char) for char in str(category_name)) + seed) % 2 == 0 else 'test'
            split_plan = {
                group_keys[0]: evaluation_split,
                group_keys[1]: 'train',
            }
            for group_key, split_name in split_plan.items():
                mask = category_group['split_group'].eq(group_key)
                for index in category_group.loc[mask].index:
                    split_by_index[index] = split_name
            continue

        if size == 3:
            split_plan = {
                group_keys[0]: 'test',
                group_keys[1]: 'validation',
                group_keys[2]: 'train',
            }
            for group_key, split_name in split_plan.items():
                mask = category_group['split_group'].eq(group_key)
                for index in category_group.loc[mask].index:
                    split_by_index[index] = split_name
            continue

        test_size = max(1, int(round(size * 0.15)))
        validation_size = max(1, int(round(size * 0.15)))

        while size - test_size - validation_size < 1 and validation_size > 0:
            validation_size -= 1
        while size - test_size - validation_size < 1 and test_size > 0:
            test_size -= 1

        test_groups = group_keys[:test_size]
        validation_groups = group_keys[test_size:test_size + validation_size]
        train_groups = group_keys[test_size + validation_size:]

        for group_key in train_groups:
            mask = category_group['split_group'].eq(group_key)
            for index in category_group.loc[mask].index:
                split_by_index[index] = 'train'
        for group_key in validation_groups:
            mask = category_group['split_group'].eq(group_key)
            for index in category_group.loc[mask].index:
                split_by_index[index] = 'validation'
        for group_key in test_groups:
            mask = category_group['split_group'].eq(group_key)
            for index in category_group.loc[mask].index:
                split_by_index[index] = 'test'

    return df.index.to_series().map(split_by_index).fillna('train')


def build_training_dataset(df, min_confidence, seed):
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
    df['split_group'] = (
        df['merchant_normalized'].fillna('unknown') + '|' +
        df['description_normalized'].fillna('') + '|' +
        df['target_category'].fillna('uncategorized')
    )
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

    training_df = df.loc[accepted_mask].copy()
    review_df = df.loc[~accepted_mask].copy()

    class_support = training_df['target_category'].value_counts()
    training_df['target_class_count'] = training_df['target_category'].map(class_support).astype(int)
    training_df['target_is_rare'] = training_df['target_class_count'].lt(5)
    training_df['dataset_split'] = assign_split(training_df, seed)
    training_df = training_df.drop(columns=['split_group'])
    review_df = review_df.drop(columns=['split_group'])

    training_columns = [
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

    review_columns = [
        'transaction_id', 'provider_transaction_id', 'user_id', 'user_label',
        'booked_date', 'direction', 'amount_signed', 'amount_abs',
        'description_original', 'description_display', 'description_normalized',
        'merchant_normalized', 'category_general', 'label_source',
        'label_confidence', 'review_reason', 'merchant_tx_count', 'merchant_active_months',
        'merchant_recurring_candidate',
    ]

    return training_df[training_columns], review_df[review_columns]


def save_dataset(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_input_dataset(input_path):
    return pd.read_csv(input_path, usecols=REQUIRED_COLUMNS)


def main():
    args = parse_args()

    print('=' * 60)
    print('PREPARACION DE DATASET DE ENTRENAMIENTO TINK')
    print('=' * 60)

    df = load_input_dataset(args.input)
    training_df, review_df = build_training_dataset(df, args.min_confidence, args.seed)

    save_dataset(training_df, args.raw_output)
    save_dataset(training_df, args.clean_output)
    save_dataset(review_df, args.review_output)

    print(f'\nEntrenamiento: {len(training_df):,} transacciones')
    print(f'Revision manual: {len(review_df):,} transacciones')
    print(f'Categorias objetivo: {training_df["target_category"].nunique():,}')
    print(f'Usuarios en entrenamiento: {training_df["user_id"].nunique():,}')
    print('\nDistribucion por split:')
    print(training_df['dataset_split'].value_counts().to_string())
    print('\nTop categorias:')
    print(training_df['target_category'].value_counts().head(10).to_string())
    print(f'\nArchivo raw: {args.raw_output}')
    print(f'Archivo clean: {args.clean_output}')
    print(f'Archivo review: {args.review_output}')


if __name__ == '__main__':
    main()