import argparse
import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / 'data' / 'raw'
CLEAN_DIR = ROOT_DIR / 'data' / 'clean'
MODELS_DIR = ROOT_DIR / '05_modelos' / 'artifacts'

DEFAULT_INPUT = CLEAN_DIR / 'transacciones_tink_model_input.csv'
DEFAULT_METRICS_OUTPUT = CLEAN_DIR / 'metricas_clasificador_categorias_tink.json'
DEFAULT_PREDICTIONS_OUTPUT = CLEAN_DIR / 'predicciones_clasificador_categorias_tink.csv'
DEFAULT_MODEL_OUTPUT = MODELS_DIR / 'clasificador_categorias_tink.pkl'
DEFAULT_CONFUSION_OUTPUT = CLEAN_DIR / 'clasificador_categorias_tink_confusion.csv'

TEXT_FEATURE = 'description_normalized'
NUMERIC_FEATURES = [
    'amount_abs',
    'booking_day_of_month',
    'text_token_count',
    'merchant_tx_count',
    'merchant_active_months',
    'merchant_total_amount',
]
CATEGORICAL_FEATURES = [
    'direction',
    'amount_bucket',
    'currency_code',
    'merchant_normalized',
    'booking_weekday',
    'language_hint',
    'account_type',
    'customer_segment',
]
BOOLEAN_FEATURES = [
    'is_weekend',
    'merchant_known',
    'merchant_recurring_candidate',
    'is_income',
    'is_expense',
]
TARGET_COLUMN = 'target_category'
SPLIT_COLUMN = 'dataset_split'

REQUIRED_COLUMNS = [
    TEXT_FEATURE,
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
    *BOOLEAN_FEATURES,
    TARGET_COLUMN,
    SPLIT_COLUMN,
    'target_class_count',
    'transaction_id',
    'user_id',
    'description_original',
    'merchant_normalized',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Entrena un clasificador base de categorias sobre transacciones Tink.'
    )
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help='CSV de capa 3 listo para entrenamiento.')
    parser.add_argument('--metrics-output', type=Path, default=DEFAULT_METRICS_OUTPUT,
                        help='JSON de metricas y configuracion del modelo.')
    parser.add_argument('--predictions-output', type=Path, default=DEFAULT_PREDICTIONS_OUTPUT,
                        help='CSV con predicciones sobre validation y test.')
    parser.add_argument('--model-output', type=Path, default=DEFAULT_MODEL_OUTPUT,
                        help='Pipeline serializado entrenado.')
    parser.add_argument('--confusion-output', type=Path, default=DEFAULT_CONFUSION_OUTPUT,
                        help='CSV de matriz de confusion del test set.')
    parser.add_argument('--min-class-support', type=int, default=5,
                        help='Soporte minimo por clase para incluirla en entrenamiento.')
    parser.add_argument('--max-text-features', type=int, default=500,
                        help='Maximo de features TF-IDF para descripcion normalizada.')
    parser.add_argument('--min-merchant-support', type=int, default=20,
                        help='Soporte minimo para conservar merchants como categoria explicita.')
    parser.add_argument('--chunk-size', type=int, default=50000,
                        help='Numero de filas por chunk al procesar el CSV completo.')
    parser.add_argument('--categorical-hash-features', type=int, default=2**15,
                        help='Dimensionalidad del hashing para variables categoricas y booleanas.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Numero de pasadas sobre el conjunto de entrenamiento.')
    parser.add_argument('--progress-every', type=int, default=10,
                        help='Numero de chunks entre mensajes de progreso.')
    return parser.parse_args()


def iter_dataset_chunks(input_path, usecols, chunk_size):
    return pd.read_csv(input_path, usecols=usecols, chunksize=chunk_size)


def prepare_chunk(df, frequent_merchants):
    df = df.copy()
    for column in BOOLEAN_FEATURES:
        df[column] = df[column].fillna(False).astype(int)
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0.0)
    df['booking_day_of_month'] = pd.to_numeric(df['booking_day_of_month'], errors='coerce').fillna(0).astype(float)
    df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna('')
    df['description_original'] = df['description_original'].fillna('')
    df['merchant_normalized'] = df['merchant_normalized'].fillna('unknown')
    df['merchant_normalized'] = df['merchant_normalized'].where(
        df['merchant_normalized'].isin(frequent_merchants),
        'other_merchant',
    )
    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].fillna('unknown').astype(str)
    return df


def build_feature_extractors(text_features, categorical_hash_features):
    text_vectorizer = HashingVectorizer(
        n_features=text_features,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm='l2',
    )
    categorical_hasher = FeatureHasher(
        n_features=categorical_hash_features,
        input_type='dict',
        alternate_sign=False,
    )
    numeric_scaler = StandardScaler()
    classifier = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=0.0001,
        learning_rate='optimal',
        random_state=42,
    )
    return text_vectorizer, categorical_hasher, numeric_scaler, classifier


def build_categorical_records(df):
    records = []
    for row in df[CATEGORICAL_FEATURES + BOOLEAN_FEATURES].itertuples(index=False, name=None):
        feature_dict = {}
        for index, feature in enumerate(CATEGORICAL_FEATURES):
            feature_dict[f'{feature}={row[index]}'] = 1.0
        for offset, feature in enumerate(BOOLEAN_FEATURES, start=len(CATEGORICAL_FEATURES)):
            feature_dict[feature] = float(row[offset])
        records.append(feature_dict)
    return records


def build_sparse_matrix(df, text_vectorizer, categorical_hasher, numeric_scaler):
    text_matrix = text_vectorizer.transform(df[TEXT_FEATURE].astype(str).tolist())
    categorical_matrix = categorical_hasher.transform(build_categorical_records(df))
    numeric_matrix = numeric_scaler.transform(df[NUMERIC_FEATURES].to_numpy(dtype=float))
    numeric_matrix = csr_matrix(numeric_matrix)
    return hstack([text_matrix, numeric_matrix, categorical_matrix], format='csr')


def collect_training_metadata(input_path, min_class_support, min_merchant_support, chunk_size, progress_every):
    metadata_columns = [TARGET_COLUMN, SPLIT_COLUMN, 'target_class_count', 'merchant_normalized']
    class_names = set()
    split_counts = Counter()
    merchant_counts = Counter()

    for chunk_index, chunk in enumerate(iter_dataset_chunks(input_path, metadata_columns, chunk_size), start=1):
        chunk['target_class_count'] = pd.to_numeric(chunk['target_class_count'], errors='coerce').fillna(0)
        filtered = chunk.loc[chunk['target_class_count'].ge(min_class_support)].copy()
        if filtered.empty:
            continue

        split_counts.update(filtered[SPLIT_COLUMN].fillna('unknown').tolist())
        train_rows = filtered.loc[filtered[SPLIT_COLUMN] == 'train']
        if not train_rows.empty:
            class_names.update(train_rows[TARGET_COLUMN].dropna().astype(str).tolist())
            merchant_counts.update(train_rows['merchant_normalized'].fillna('unknown').astype(str).tolist())
        if chunk_index % progress_every == 0:
            print(f'Metadata procesada: {chunk_index} chunks...', flush=True)

    frequent_merchants = {
        merchant for merchant, count in merchant_counts.items()
        if count >= min_merchant_support
    }
    frequent_merchants.add('unknown')

    return sorted(class_names), split_counts, frequent_merchants


def iter_filtered_subset_chunks(input_path, subset_name, min_class_support, chunk_size, frequent_merchants):
    for chunk in iter_dataset_chunks(input_path, REQUIRED_COLUMNS, chunk_size):
        chunk['target_class_count'] = pd.to_numeric(chunk['target_class_count'], errors='coerce').fillna(0)
        filtered = chunk.loc[
            chunk['target_class_count'].ge(min_class_support) &
            chunk[SPLIT_COLUMN].eq(subset_name)
        ].copy()
        if filtered.empty:
            continue
        yield prepare_chunk(filtered, frequent_merchants)


def fit_numeric_scaler(input_path, min_class_support, chunk_size, frequent_merchants, progress_every):
    scaler = StandardScaler()
    fitted = False

    for chunk_index, chunk in enumerate(
        iter_filtered_subset_chunks(input_path, 'train', min_class_support, chunk_size, frequent_merchants),
        start=1,
    ):
        scaler.partial_fit(chunk[NUMERIC_FEATURES].to_numpy(dtype=float))
        fitted = True
        if chunk_index % progress_every == 0:
            print(f'Scaler ajustado: {chunk_index} chunks de train...', flush=True)

    if not fitted:
        raise ValueError('No hay filas de entrenamiento tras aplicar el filtro de soporte minimo.')

    return scaler


def train_incremental_model(input_path, min_class_support, chunk_size, frequent_merchants,
                            text_vectorizer, categorical_hasher, numeric_scaler,
                            classifier, classes, epochs, progress_every):
    fitted = False

    for epoch in range(epochs):
        print(f'Entrenando epoch {epoch + 1}/{epochs}...', flush=True)
        for chunk_index, chunk in enumerate(
            iter_filtered_subset_chunks(input_path, 'train', min_class_support, chunk_size, frequent_merchants),
            start=1,
        ):
            X_chunk = build_sparse_matrix(chunk, text_vectorizer, categorical_hasher, numeric_scaler)
            y_chunk = chunk[TARGET_COLUMN].astype(str).to_numpy()
            if not fitted:
                classifier.partial_fit(X_chunk, y_chunk, classes=np.array(classes, dtype=object))
                fitted = True
            else:
                classifier.partial_fit(X_chunk, y_chunk)
            if chunk_index % progress_every == 0:
                print(f'Epoch {epoch + 1}: entrenados {chunk_index} chunks...', flush=True)

    if not fitted:
        raise ValueError('No hay filas de entrenamiento tras aplicar el filtro de soporte minimo.')

    return classifier


def evaluate_subset_stream(model, input_path, subset_name, min_class_support, chunk_size,
                           frequent_merchants, text_vectorizer, categorical_hasher,
                           numeric_scaler, predictions_output, write_header, progress_every):
    y_true = []
    y_pred = []

    for chunk_index, chunk in enumerate(
        iter_filtered_subset_chunks(input_path, subset_name, min_class_support, chunk_size, frequent_merchants),
        start=1,
    ):
        X_subset = build_sparse_matrix(chunk, text_vectorizer, categorical_hasher, numeric_scaler)
        predicted = model.predict(X_subset)
        actual = chunk[TARGET_COLUMN].astype(str).to_numpy()

        y_true.extend(actual.tolist())
        y_pred.extend(predicted.tolist())

        predictions_df = chunk[[
            'transaction_id', 'user_id', 'description_original', 'merchant_normalized',
            'amount_abs', TARGET_COLUMN,
        ]].copy()
        predictions_df['subset'] = subset_name
        predictions_df['predicted_category'] = predicted
        predictions_df['is_correct'] = predictions_df[TARGET_COLUMN].eq(predictions_df['predicted_category'])
        predictions_df.to_csv(predictions_output, index=False, mode='w' if write_header else 'a', header=write_header)
        write_header = False
        if chunk_index % progress_every == 0:
            print(f'Evaluacion {subset_name}: procesados {chunk_index} chunks...', flush=True)

    if not y_true:
        raise ValueError(f'Faltan filas en {subset_name}; revisa el particionado de capa 3.')

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return accuracy, report, y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, output_path):
    labels = sorted(pd.Series(y_true).unique())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_df.to_csv(output_path)


def save_json(payload, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    print('=' * 60)
    print('CLASIFICADOR DE CATEGORIAS TINK')
    print('=' * 60)

    classes, split_counts, frequent_merchants = collect_training_metadata(
        args.input,
        args.min_class_support,
        args.min_merchant_support,
        args.chunk_size,
        args.progress_every,
    )
    if not classes:
        raise ValueError('No hay clases validas en train tras aplicar el filtro de soporte minimo.')
    if split_counts.get('validation', 0) == 0 or split_counts.get('test', 0) == 0:
        raise ValueError('Faltan filas en validation o test; revisa el particionado de capa 3.')

    text_vectorizer, categorical_hasher, _, classifier = build_feature_extractors(
        args.max_text_features,
        args.categorical_hash_features,
    )
    numeric_scaler = fit_numeric_scaler(
        args.input,
        args.min_class_support,
        args.chunk_size,
        frequent_merchants,
        args.progress_every,
    )
    model = train_incremental_model(
        args.input,
        args.min_class_support,
        args.chunk_size,
        frequent_merchants,
        text_vectorizer,
        categorical_hasher,
        numeric_scaler,
        classifier,
        classes,
        args.epochs,
        args.progress_every,
    )

    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    if args.predictions_output.exists():
        args.predictions_output.unlink()

    validation_accuracy, validation_report, _, _ = evaluate_subset_stream(
        model,
        args.input,
        'validation',
        args.min_class_support,
        args.chunk_size,
        frequent_merchants,
        text_vectorizer,
        categorical_hasher,
        numeric_scaler,
        args.predictions_output,
        True,
        args.progress_every,
    )
    test_accuracy, test_report, test_true, test_pred = evaluate_subset_stream(
        model,
        args.input,
        'test',
        args.min_class_support,
        args.chunk_size,
        frequent_merchants,
        text_vectorizer,
        categorical_hasher,
        numeric_scaler,
        args.predictions_output,
        False,
        args.progress_every,
    )

    plot_confusion_matrix(test_true, test_pred, args.confusion_output)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    with args.model_output.open('wb') as file_handle:
        pickle.dump({
            'classifier': model,
            'text_vectorizer': text_vectorizer,
            'categorical_hasher': categorical_hasher,
            'numeric_scaler': numeric_scaler,
            'frequent_merchants': sorted(frequent_merchants),
            'classes': classes,
            'feature_columns': {
                'text': TEXT_FEATURE,
                'numeric': NUMERIC_FEATURES,
                'categorical': CATEGORICAL_FEATURES,
                'boolean': BOOLEAN_FEATURES,
            },
        }, file_handle)

    metrics = {
        'input_path': str(args.input),
        'min_class_support': args.min_class_support,
        'max_text_features': args.max_text_features,
        'min_merchant_support': args.min_merchant_support,
        'chunk_size': args.chunk_size,
        'categorical_hash_features': args.categorical_hash_features,
        'epochs': args.epochs,
        'rows_total': int(sum(split_counts.values())),
        'rows_train': int(split_counts.get('train', 0)),
        'rows_validation': int(split_counts.get('validation', 0)),
        'rows_test': int(split_counts.get('test', 0)),
        'classes': classes,
        'validation_accuracy': round(float(validation_accuracy), 4),
        'test_accuracy': round(float(test_accuracy), 4),
        'validation_report': validation_report,
        'test_report': test_report,
        'artifacts': {
            'model': str(args.model_output),
            'predictions': str(args.predictions_output),
            'confusion_matrix': str(args.confusion_output),
        },
    }
    save_json(metrics, args.metrics_output)

    print(f'\nFilas totales usadas: {sum(split_counts.values()):,}')
    print(
        f'Train: {split_counts.get("train", 0):,} | '
        f'Validation: {split_counts.get("validation", 0):,} | '
        f'Test: {split_counts.get("test", 0):,}'
    )
    print(f'Validation accuracy: {validation_accuracy:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')
    print(f'\nMetricas: {args.metrics_output}')
    print(f'Predicciones: {args.predictions_output}')
    print(f'Modelo: {args.model_output}')
    print(f'Matriz confusion: {args.confusion_output}')


if __name__ == '__main__':
    main()