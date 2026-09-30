import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ACCOUNTS_DIR = ROOT_DIR / 'accounts'
OUTPUT_DIR = ACCOUNTS_DIR / 'synthetic'

ACCOUNT_FILE_RE = re.compile(r'^salida list accounts_(?P<source>.+)\.txt$')
TRANSACTION_FILE_RE = re.compile(
    r'^salida list transactions_(?P<source>.+?)(?:_(?P<page>\d+))?\.txt$'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Genera payloads sinteticos tipo Tink a partir de las muestras en accounts/.'
    )
    parser.add_argument('--users', type=int, default=20,
                        help='Cantidad de usuarios sinteticos a generar.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria para reproducibilidad.')
    parser.add_argument('--max-transactions', type=int, default=800,
                        help='Maximo de transacciones por usuario sintetico.')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                        help='Carpeta de salida para los payloads generados.')
    parser.add_argument('--checkpoint-every', type=int, default=250,
                        help='Cada cuantos usuarios guardar progreso y mostrar avance.')
    return parser.parse_args()


def load_json(path):
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def amount_from_value(amount_payload):
    unscaled = int(amount_payload['value']['unscaledValue'])
    scale = int(amount_payload['value']['scale'])
    return unscaled / (10 ** scale)


def value_payload(amount, scale):
    unscaled = int(round(amount * (10 ** scale)))
    return {
        'value': {
            'unscaledValue': str(unscaled),
            'scale': str(scale),
        }
    }


def stable_hex(seed_text, size=32):
    digest = hashlib.md5(seed_text.encode('utf-8')).hexdigest()
    return digest[:size]


def stable_iban(country_code, ordinal):
    digits = ''.join(ch for ch in stable_hex(f'iban-{country_code}-{ordinal}', 28) if ch.isdigit())
    if len(digits) < 22:
        digits = (digits + '0' * 22)[:22]
    return f'{country_code}{digits[:22]}'


def account_type_localized(account_type, language):
    names = {
        'CHECKING': {
            'es': ['Cuenta corriente', 'Cuenta dia a dia', 'Cuenta principal'],
            'de': ['Girokonto', 'Alltagskonto', 'Privatkonto'],
            'generic': ['Account 1', 'Daily account', 'Primary account'],
        },
        'SAVINGS': {
            'es': ['Cuenta ahorro', 'Ahorro objetivo', 'Deposito ahorro'],
            'de': ['Sparkonto', 'Tagesgeld', 'Reservekonto'],
            'generic': ['Savings account', 'Reserve account', 'Account 2'],
        },
        'CREDIT_CARD': {
            'es': ['Tarjeta credito'],
            'de': ['Kreditkarte'],
            'generic': ['Credit card'],
        },
    }
    options = names.get(account_type, names['CHECKING'])
    return random.choice(options.get(language, options['generic']))


SALARY_PATTERNS = [
    r'\bsalary\b', r'salario', r'payroll', r'n[oó]?mina', r'haberes',
    r'\bgehalt\b', r'lohn', r'\bnomin',
]

DEFAULT_BASE_SALARY = {
    'es': 1550.0,
    'de': 2100.0,
    'generic': 1600.0,
}


def infer_salary(income_amounts, income_descriptions, language):
    """Deriva una descripcion y un importe base de nomina realistas a partir
    de las transacciones de credito observadas en la muestra real.

    income_amounts e income_descriptions estan alineados por indice (se
    construyen en el mismo bucle en build_profiles), asi que se pueden
    emparejar directamente.
    """
    pairs = list(zip(income_descriptions, income_amounts))
    salary_pairs = [
        (desc, amount) for desc, amount in pairs
        if any(re.search(pattern, desc.lower()) for pattern in SALARY_PATTERNS)
    ]

    if salary_pairs:
        amounts = sorted(amount for _, amount in salary_pairs)
        base_salary = amounts[len(amounts) // 2]
        description = Counter(desc for desc, _ in salary_pairs).most_common(1)[0][0]
        return description, base_salary

    if income_amounts:
        amounts = sorted(income_amounts)
        base_salary = amounts[len(amounts) // 2]
        description = 'Salary' if language != 'es' else 'Nomina'
        return description, max(base_salary, 400.0)

    description = 'Salary' if language != 'es' else 'Nomina'
    return description, DEFAULT_BASE_SALARY.get(language, DEFAULT_BASE_SALARY['generic'])


MIN_EXPENSE_TX_PER_MONTH = 4
MAX_EXPENSE_TX_PER_MONTH = 35
FALLBACK_EXPENSE_TX_PER_MONTH = 12.0


def months_between(start_date, end_date):
    months = []
    cursor = start_date.replace(day=1)
    end_cursor = end_date.replace(day=1)
    while cursor <= end_cursor:
        months.append(cursor)
        if cursor.month == 12:
            cursor = cursor.replace(year=cursor.year + 1, month=1)
        else:
            cursor = cursor.replace(month=cursor.month + 1)
    return months


def infer_language(descriptions):
    text = ' '.join(descriptions).lower()
    spanish_markers = ['manutención', 'abono', 'deposito', 'haberes', 'cuenta']
    german_markers = ['miete', 'zinserträge', 'sparkonto', 'girokonto']
    if any(marker in text for marker in spanish_markers):
        return 'es'
    if any(marker in text for marker in german_markers):
        return 'de'
    return 'generic'


def load_samples(accounts_dir):
    account_payloads = {}
    transaction_payloads = defaultdict(list)

    for path in sorted(accounts_dir.glob('*.txt')):
        account_match = ACCOUNT_FILE_RE.match(path.name)
        transaction_match = TRANSACTION_FILE_RE.match(path.name)
        if account_match:
            account_payloads[account_match.group('source')] = load_json(path)
        elif transaction_match:
            source = transaction_match.group('source')
            page = int(transaction_match.group('page') or 1)
            transaction_payloads[source].append((page, load_json(path)))

    samples = []
    for source, payload in account_payloads.items():
        pages = [payload for _, payload in sorted(transaction_payloads.get(source, []), key=lambda item: item[0])]
        sample = {
            'source': source,
            'accounts': payload.get('accounts', []),
            'transaction_pages': pages,
            'transactions': [
                transaction
                for page in pages
                for transaction in page.get('transactions', [])
            ],
        }
        samples.append(sample)
    return samples


def build_profiles(samples):
    profiles = []

    for sample in samples:
        transactions = sample['transactions']
        descriptions = [tx['descriptions']['original'] for tx in transactions if 'descriptions' in tx]
        language = infer_language(descriptions)

        per_account = defaultdict(list)
        for tx in transactions:
            per_account[tx['accountId']].append(tx)

        income = []
        expense = []
        scale_counter = Counter()
        currencies = Counter()
        statuses = Counter()
        merchants_positive = []
        merchants_negative = []
        booked_dates = []

        for tx in transactions:
            amount_payload = tx['amount']
            amount = amount_from_value(amount_payload)
            scale_counter[int(amount_payload['value']['scale'])] += 1
            currencies[amount_payload['currencyCode']] += 1
            statuses[tx.get('status', 'BOOKED')] += 1
            booked_dates.append(datetime.strptime(tx['dates']['booked'], '%Y-%m-%d').date())
            if amount >= 0:
                income.append(amount)
                merchants_positive.append(tx['descriptions']['original'])
            else:
                expense.append(abs(amount))
                merchants_negative.append(tx['descriptions']['original'])

        account_templates = []
        for account in sample['accounts']:
            booked_amount = account['balances']['booked']['amount']
            available_amount = account['balances'].get('available', {}).get('amount', booked_amount)
            booked = amount_from_value(booked_amount)
            available = amount_from_value(available_amount)
            account_templates.append({
                'type': account.get('type', 'CHECKING'),
                'customerSegment': account.get('customerSegment', 'PERSONAL'),
                'financialInstitutionId': account.get('financialInstitutionId', ''),
                'balance_booked': booked,
                'balance_available': available,
                'currencyCode': account['balances']['booked']['amount']['currencyCode'],
                'refresh': account.get('dates', {}).get('lastRefreshed', '2026-07-05T00:00:00Z'),
                'tx_count': len(per_account.get(account['id'], [])),
            })

        start_date = min(booked_dates) if booked_dates else datetime(2026, 1, 1).date()
        end_date = max(booked_dates) if booked_dates else datetime(2026, 7, 1).date()

        salary_description, base_salary = infer_salary(income, merchants_positive, language)

        # Frecuencia media de gasto por mes. Las muestras reales cubren
        # ventanas de observacion muy dispares (2 a 52 meses) y algunas no
        # registran transacciones de debito en absoluto, asi que un
        # promedio crudo por perfil no es fiable: se descarta cuando cae
        # fuera de un rango plausible de actividad bancaria personal y se
        # usa en su lugar un valor de referencia (fallback), para no dejar
        # meses sinteticos sin ningun gasto o con cientos de ellos.
        real_months = months_between(start_date, end_date)
        raw_avg_expense = len(expense) / max(1, len(real_months))
        if MIN_EXPENSE_TX_PER_MONTH <= raw_avg_expense <= MAX_EXPENSE_TX_PER_MONTH:
            avg_expense_tx_per_month = raw_avg_expense
        else:
            avg_expense_tx_per_month = FALLBACK_EXPENSE_TX_PER_MONTH

        # Ingresos "secundarios" (intereses, transferencias puntuales, etc.):
        # se excluyen las entradas que ya se usan para modelar la nomina
        # mensual, para no duplicar importes de tipo salario como relleno.
        secondary_income_descriptions = []
        secondary_income_amounts = []
        for desc, amount in zip(merchants_positive, income):
            if any(re.search(pattern, desc.lower()) for pattern in SALARY_PATTERNS):
                continue
            secondary_income_descriptions.append(desc)
            secondary_income_amounts.append(amount)

        profiles.append({
            'source': sample['source'],
            'language': language,
            'account_templates': account_templates,
            'income_amounts': income,
            'expense_amounts': expense,
            'income_descriptions': merchants_positive,
            'expense_descriptions': merchants_negative,
            'secondary_income_descriptions': secondary_income_descriptions,
            'secondary_income_amounts': secondary_income_amounts,
            'scale_weights': scale_counter,
            'currency_weights': currencies,
            'status_weights': statuses,
            'start_date': start_date,
            'end_date': end_date,
            'transactions_per_user': len(transactions),
            'salary_description': salary_description,
            'base_salary': base_salary,
            'avg_expense_tx_per_month': avg_expense_tx_per_month,
        })

    return [profile for profile in profiles if profile['account_templates']]


def weighted_choice(counter):
    items = list(counter.items())
    values = [value for value, _ in items]
    weights = [weight for _, weight in items]
    return random.choices(values, weights=weights, k=1)[0]


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def jitter_amount(base_amount, positive):
    variance = 0.35 if positive else 0.45
    factor = clamp(random.gauss(1.0, variance), 0.35, 2.2)
    amount = round(base_amount * factor, 2)
    if positive:
        return max(amount, 10.0)
    return max(amount, 2.5)


def jitter_salary(base_salary):
    """Variacion pequena y realista para la nomina mensual (p.ej. horas
    extra o pequenos ajustes), sin el efecto de cola larga del amplificador
    aleatorio que se usaba antes para todos los ingresos."""
    factor = clamp(random.gauss(1.0, 0.04), 0.9, 1.12)
    return round(max(base_salary, 1.0) * factor, 2)


def jitter_secondary_income(base_amount):
    """Ingresos secundarios ocasionales (intereses, transferencias
    puntuales): variacion moderada, sin el multiplicador de hasta x2.8 que
    provocaba salarios mensuales acumulados irreales."""
    factor = clamp(random.gauss(1.0, 0.25), 0.5, 1.5)
    return round(max(base_amount * factor, 5.0), 2)


def make_account_payload(profile, synthetic_user_id):
    country_code = 'ES' if profile['language'] == 'es' else 'DE'
    payload_accounts = []
    account_id_map = {}
    transaction_targets = {}

    for index, template in enumerate(profile['account_templates'], start=1):
        account_key = f'{synthetic_user_id}-account-{index}'
        account_id = stable_hex(account_key)
        account_id_map[index] = account_id

        booked_balance = round(jitter_amount(abs(template['balance_booked']) or 250.0, template['balance_booked'] >= 0), 2)
        if template['balance_booked'] < 0:
            booked_balance *= -1

        available_balance = booked_balance
        scale = weighted_choice(Counter({2: 8, 1: 2}))
        iban = stable_iban(country_code, synthetic_user_id * 10 + index)

        transaction_targets[account_id] = max(8, int(template['tx_count'] * random.uniform(0.7, 1.4)))

        payload_accounts.append({
            'id': account_id,
            'name': account_type_localized(template['type'], profile['language']),
            'type': template['type'],
            'balances': {
                'booked': {
                    'amount': {
                        **value_payload(booked_balance, scale),
                        'currencyCode': template['currencyCode'],
                    }
                },
                'available': {
                    'amount': {
                        **value_payload(available_balance, scale),
                        'currencyCode': template['currencyCode'],
                    }
                }
            },
            'identifiers': {
                'iban': {
                    'iban': iban,
                    'bban': iban[4:],
                },
                'financialInstitution': {
                    'accountNumber': iban,
                    'referenceNumbers': {},
                }
            },
            'dates': {
                'lastRefreshed': template['refresh'],
            },
            'financialInstitutionId': template['financialInstitutionId'],
            'customerSegment': template['customerSegment'],
        })

    return {
        'accounts': payload_accounts,
        'nextPageToken': '',
    }, transaction_targets


def build_transaction_payload(profile, synthetic_user_id, tx_index, account_id, signed_amount, description, booked_date):
    scale = weighted_choice(profile['scale_weights'] or Counter({2: 1}))
    currency = weighted_choice(profile['currency_weights'] or Counter({'EUR': 1}))
    status = weighted_choice(profile['status_weights'] or Counter({'BOOKED': 1}))
    tx_id = stable_hex(f'{synthetic_user_id}-{account_id}-{tx_index}')
    provider_tx_id = str(1000 + synthetic_user_id * 1000 + tx_index)

    return {
        'id': tx_id,
        'accountId': account_id,
        'amount': {
            **value_payload(signed_amount, scale),
            'currencyCode': currency,
        },
        'descriptions': {
            'original': description,
            'display': description.title(),
        },
        'dates': {
            'booked': booked_date.strftime('%Y-%m-%d'),
        },
        'identifiers': {
            'providerTransactionId': provider_tx_id,
        },
        'types': {
            'type': 'DEFAULT',
        },
        'status': status,
        'providerMutability': 'MUTABILITY_UNDEFINED',
    }


def random_day_in_month(month_start, profile):
    if month_start.month == 12:
        next_month = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month = month_start.replace(month=month_start.month + 1)
    days_in_month = (next_month - timedelta(days=1)).day
    day = random.randint(1, days_in_month)
    booked_date = month_start.replace(day=day)
    return min(max(booked_date, profile['start_date']), profile['end_date'])


def make_transaction(profile, synthetic_user_id, tx_index, account_id, booked_date):
    """Genera una transaccion de "relleno" para el mes/dia indicados: un
    gasto habitual, o de forma ocasional (8%) un ingreso secundario pequeno
    (interes, transferencia puntual). La nomina mensual se genera aparte,
    en make_salary_transaction, para evitar que el salario del mes se
    infle con multiples creditos aleatorios."""
    positive = random.random() < 0.08
    description_pool = profile['secondary_income_descriptions'] if positive else profile['expense_descriptions']
    amount_pool = profile['secondary_income_amounts'] if positive else profile['expense_amounts']
    description = random.choice(description_pool) if description_pool else ('Interest' if positive else 'Card payment')
    amount = random.choice(amount_pool) if amount_pool else (25.0 if positive else 34.5)

    if positive:
        # Tope de seguridad: ademas del filtro por descripcion, ningun
        # ingreso "secundario" de relleno puede superar una fraccion de la
        # nomina base, para evitar que una transferencia puntual grande
        # residual en la muestra real distorsione el mes.
        amount = min(amount, profile['base_salary'] * 0.6)

    amount = jitter_secondary_income(amount) if positive else jitter_amount(amount, positive=False)
    signed_amount = amount if positive else -amount

    return build_transaction_payload(
        profile, synthetic_user_id, tx_index, account_id, signed_amount, description, booked_date,
    )


def make_salary_transaction(profile, synthetic_user_id, tx_index, account_id, month_start):
    """Genera exactamente un abono de nomina para el mes indicado, con un
    dia de pago estable (24-28) y una variacion pequena y realista."""
    if month_start.month == 12:
        next_month = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month = month_start.replace(month=month_start.month + 1)
    days_in_month = (next_month - timedelta(days=1)).day
    pay_day = min(24 + (synthetic_user_id % 5), days_in_month)
    booked_date = month_start.replace(day=pay_day)
    booked_date = min(max(booked_date, profile['start_date']), profile['end_date'])

    amount = jitter_salary(profile['base_salary'])

    return build_transaction_payload(
        profile, synthetic_user_id, tx_index, account_id, amount, profile['salary_description'], booked_date,
    )


def make_transaction_pages(profile, synthetic_user_id, account_payload, transaction_targets, max_transactions):
    """Genera las transacciones mes a mes: un abono de nomina garantizado
    (make_salary_transaction) mas un numero de gastos calibrado sobre la
    actividad real observada (avg_expense_tx_per_month), en lugar de
    repartir un presupuesto total fijo de transacciones entre nomina y
    gastos. Ese reparto por presupuesto dejaba, con muchos meses de
    historial, la mayoria de los meses con una unica transaccion (solo la
    nomina) y una tasa de ahorro cercana al 100%, muy alejada de un
    comportamiento financiero real."""
    accounts = account_payload['accounts']
    account_ids = [account['id'] for account in accounts]
    if not account_ids:
        return []

    # Cuenta principal para el abono de nomina: preferimos una CHECKING si
    # existe, si no la primera cuenta disponible.
    primary_account = next((acc['id'] for acc in accounts if acc.get('type') == 'CHECKING'), account_ids[0])
    account_weights = [max(1, transaction_targets.get(acc_id, 1)) for acc_id in account_ids]

    months = months_between(profile['start_date'], profile['end_date'])
    transactions = []
    tx_index = 1

    for month_start in months:
        transactions.append(make_salary_transaction(profile, synthetic_user_id, tx_index, primary_account, month_start))
        tx_index += 1

        expense_count = max(1, round(profile['avg_expense_tx_per_month'] * random.uniform(0.75, 1.25)))
        for _ in range(expense_count):
            account_id = random.choices(account_ids, weights=account_weights, k=1)[0]
            booked_date = random_day_in_month(month_start, profile)
            transactions.append(make_transaction(profile, synthetic_user_id, tx_index, account_id, booked_date))
            tx_index += 1

        if len(transactions) >= max_transactions:
            break

    transactions = transactions[:max_transactions]
    transactions.sort(key=lambda tx: (tx['dates']['booked'], tx['id']), reverse=True)

    pages = []
    page_size = 100
    for page_index, start in enumerate(range(0, len(transactions), page_size), start=1):
        chunk = transactions[start:start + page_size]
        next_page_token = '' if start + page_size >= len(transactions) else stable_hex(
            f'{synthetic_user_id}-page-{page_index}-{chunk[-1]["id"]}',
            size=64,
        )
        pages.append({
            'transactions': chunk,
            'nextPageToken': next_page_token,
        })

    return pages


def write_payload(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    with temp_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)
        handle.write('\n')
    temp_path.replace(path)


def write_manifest(manifest_path, manifest):
    write_payload(manifest_path, {
        'generated_at': datetime.now(UTC).isoformat(),
        'users': manifest,
    })


def clear_output_dir(output_dir):
    if not output_dir.exists():
        return
    for path in output_dir.glob('*'):
        if path.is_file() and path.suffix in {'.txt', '.json'}:
            path.unlink()


def generate(args):
    random.seed(args.seed)
    samples = load_samples(ACCOUNTS_DIR)
    profiles = build_profiles(samples)
    if not profiles:
        raise RuntimeError('No se encontraron muestras validas en accounts/.')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_dir(args.output_dir)
    manifest = []
    manifest_path = args.output_dir / 'manifest.json'
    checkpoint_every = max(1, args.checkpoint_every)

    try:
        for synthetic_user_id in range(1, args.users + 1):
            profile = random.choice(profiles)
            account_payload, transaction_targets = make_account_payload(profile, synthetic_user_id)
            transaction_pages = make_transaction_pages(
                profile,
                synthetic_user_id,
                account_payload,
                transaction_targets,
                args.max_transactions,
            )

            source_label = f'synthetic_u{synthetic_user_id:05d}'
            account_path = args.output_dir / f'salida list accounts_{source_label}.txt'
            write_payload(account_path, account_payload)

            tx_paths = []
            for page_index, page_payload in enumerate(transaction_pages, start=1):
                suffix = '' if len(transaction_pages) == 1 else f'_{page_index}'
                tx_path = args.output_dir / f'salida list transactions_{source_label}{suffix}.txt'
                write_payload(tx_path, page_payload)
                tx_paths.append(str(tx_path.relative_to(ROOT_DIR)))

            manifest.append({
                'synthetic_user': source_label,
                'profile_source': profile['source'],
                'language_hint': profile['language'],
                'accounts_file': str(account_path.relative_to(ROOT_DIR)),
                'transaction_files': tx_paths,
                'accounts_count': len(account_payload['accounts']),
                'transactions_count': sum(len(page['transactions']) for page in transaction_pages),
            })

            if synthetic_user_id % checkpoint_every == 0 or synthetic_user_id == args.users:
                write_manifest(manifest_path, manifest)
                print(
                    f'Progreso: {synthetic_user_id}/{args.users} usuarios generados',
                    flush=True,
                )
    finally:
        if manifest:
            write_manifest(manifest_path, manifest)

    print(f'Usuarios sinteticos generados: {len(manifest)}')
    print(f'Carpeta de salida: {args.output_dir}')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    generate(parse_args())