#!/usr/bin/env python3
"""
HW03: Probability & Language Models
CIS 5300: Natural Language Processing
"""

import math
import random
import os
from collections import Counter, defaultdict

# ============================================================
# Section 0: Setup helpers
# ============================================================

def start_pad(n):
    """Returns a padding string of length n (tilde characters)."""
    return '~' * n


def create_ngram_model(model_class, path, n=2, k=0):
    """Creates an n-gram model trained on the entire file as one string."""
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model


def create_ngram_model_lines(model_class, path, n=2, k=0):
    """Creates an n-gram model trained line-by-line on a file."""
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model


# ============================================================
# Section 1: N-gram Basics
# ============================================================

def ngrams(n, text):
    """Generate n-grams from text with start padding.

    Each n-gram is a tuple (context, char) where context is a string
    of length n consisting of the n characters preceding char in the
    padded text.

    The text is padded with n '~' characters at the beginning so that
    even the first character has a full context.

    Special case: when n=0, all contexts are the empty string ''.

    Args:
        n: The n-gram order (context length). Must be >= 0.
        text: The input string.

    Returns:
        A list of (context, char) tuples, one per character in text.
    """
    if not text:
        return []
    padded = start_pad(n) + text
    result = []
    for i in range(len(text)):
        context = padded[i: i + n]
        char = padded[i + n]
        result.append((context, char))
    return result


# ============================================================
# Section 1-2: NgramModel
# ============================================================

class NgramModel(object):
    """A character-level n-gram language model with add-k smoothing."""

    def __init__(self, n, k=0):
        """Initialize the n-gram model.

        Args:
            n: The n-gram order (context length).
            k: Smoothing parameter for add-k smoothing (default 0 = no smoothing).
        """
        self.n = n
        self.k = k
        self.vocab = set()
        # contexts[ctx] = {'count': int, 'chars': Counter}
        self.contexts = {}

    def get_vocab(self):
        """Returns the set of all characters seen by this model."""
        return self.vocab

    def get_context(self):
        """Returns the contexts dictionary. (Do not modify.)"""
        return self.contexts

    def update(self, text):
        """Process text and update internal n-gram counts.

        Extract n-grams from text, update context counts and
        character-given-context counts, and add new characters to vocab.

        Args:
            text: A string to learn from.
        """
        for char in text:
            self.vocab.add(char)
        for context, char in ngrams(self.n, text):
            if context not in self.contexts:
                self.contexts[context] = {'count': 0, 'chars': Counter()}
            self.contexts[context]['count'] += 1
            self.contexts[context]['chars'][char] += 1

    def prob(self, context, char):
        """Return P(char | context).

        When k=0 (no smoothing):
            - Unseen context: return 1/|V|
            - Seen context, unseen char: return 0
            - Otherwise: MLE count(context, char) / count(context)

        When k>0 (add-k smoothing):
            P = (count(context, char) + k) / (count(context) + k * |V|)
            - Unseen context: treat count(context) as 0

        Args:
            context: A string of length n (the preceding characters).
            char: The character whose probability to compute.

        Returns:
            A float probability in [0, 1].
        """
        V = len(self.vocab)
        if V == 0:
            return 0.0

        if self.k == 0:
            if context not in self.contexts:
                return 1.0 / V
            ctx_info = self.contexts[context]
            total = ctx_info['count']
            char_count = ctx_info['chars'].get(char, 0)
            return char_count / total
        else:
            # add-k smoothing
            if context not in self.contexts:
                ctx_count = 0
                char_count = 0
            else:
                ctx_info = self.contexts[context]
                ctx_count = ctx_info['count']
                char_count = ctx_info['chars'].get(char, 0)
            return (char_count + self.k) / (ctx_count + self.k * V)

    def random_char(self, context):
        """Sample a random character according to P(char | context).

        Uses inverse transform sampling over the sorted vocabulary.
        """
        vocab_sorted = sorted(self.vocab)
        r = random.random()
        cumulative = 0.0
        for v in vocab_sorted:
            cumulative += self.prob(context, v)
            if r < cumulative:
                return v
        # fallback (floating point edge)
        return vocab_sorted[-1]

    def random_text(self, length):
        """Generate random text of the given length.

        Starts with n '~' padding characters as the initial context,
        then repeatedly samples characters updating the context window.
        """
        context = start_pad(self.n)
        result = []
        for _ in range(length):
            char = self.random_char(context[-self.n:] if self.n > 0 else '')
            result.append(char)
            context += char
        return ''.join(result)

    def perplexity(self, text):
        """Compute the perplexity of text under this model.

        PP(W) = exp( (1/N) * sum_i -log P(w_i | context_i) )

        Returns float('inf') if any probability is 0.
        """
        if not text:
            return float('inf')
        log_sum = 0.0
        n_chars = 0
        for context, char in ngrams(self.n, text):
            p = self.prob(context, char)
            if p == 0:
                return float('inf')
            log_sum += math.log(p)
            n_chars += 1
        if n_chars == 0:
            return float('inf')
        return math.exp(-log_sum / n_chars)


# ============================================================
# Section 2.4: Probability Validation
# ============================================================

def validate_probabilities(model, contexts_to_check, tolerance=1e-6):
    """Verify that P(c|context) sums to 1.0 for each context.

    Args:
        model: An NgramModel instance.
        contexts_to_check: A list of context strings to validate.
        tolerance: How close to 1.0 the sum must be (default 1e-6).

    Returns:
        True if all contexts pass validation, False otherwise.
        Prints diagnostic information for any failures.
    """
    all_passed = True
    vocab = model.get_vocab()
    for ctx in contexts_to_check:
        total = 0.0
        has_negative = False
        for c in vocab:
            p = model.prob(ctx, c)
            if p < 0:
                has_negative = True
            total += p
        if has_negative:
            print(f"FAIL: negative probability found for context '{ctx}'")
            all_passed = False
        elif abs(total - 1.0) > tolerance:
            print(f"FAIL: P(*|'{ctx}') sums to {total:.8f}, expected 1.0")
            all_passed = False
    return all_passed


# ============================================================
# Section 3: Interpolation
# ============================================================

class NgramModelWithInterpolation(NgramModel):
    """An n-gram model that interpolates between orders 0 through n."""

    def __init__(self, n, k):
        """Initialize interpolated model with n+1 component models.

        Args:
            n: Highest n-gram order.
            k: Smoothing parameter for component models.
        """
        self.n = n
        self.k = k
        self.vocab = set()
        self.contexts = {}
        # Create n+1 component models: orders 0, 1, ..., n
        self.models = [NgramModel(i, k) for i in range(n + 1)]
        # Equal weights by default
        weight = 1.0 / (n + 1)
        self.lambdas = [weight] * (n + 1)

    def get_vocab(self):
        """Returns the vocabulary (union across all component models)."""
        vocab = set()
        for m in self.models:
            vocab |= m.get_vocab()
        return vocab

    def update(self, text):
        """Update all component models with the same text."""
        for m in self.models:
            m.update(text)

    def prob(self, context, char):
        """Return interpolated probability.

        P_interp(char|context) = sum_i lambda_i * P_i(char|context_i)

        lambdas[0] weights the highest-order model (order n),
        lambdas[-1] weights the unigram model (order 0).

        context_i is the last i characters of the full context.
        """
        total = 0.0
        for i, (lam, model) in enumerate(zip(self.lambdas, reversed(self.models))):
            # reversed(self.models): index 0 = order n (highest), index -1 = order 0
            # model.n = n - i when iterating reversed
            order = model.n
            ctx_i = context[-order:] if order > 0 else ''
            total += lam * model.prob(ctx_i, char)
        return total

    def set_lambdas(self, lambdas):
        """Set the interpolation weights.

        Args:
            lambdas: A list of n+1 weights that must sum to 1.0.
                     lambdas[0] is the weight for the highest-order model,
                     lambdas[-1] is the weight for the unigram model.

        Raises:
            ValueError: If the sum of the lambdas is not equal to 1.0.
        """
        if abs(sum(lambdas) - 1.0) > 1e-6:
            raise ValueError(f"Lambdas must sum to 1.0, got {sum(lambdas)}")
        if len(lambdas) != len(self.models):
            raise ValueError(f"Expected {len(self.models)} lambdas, got {len(lambdas)}")
        self.lambdas = list(lambdas)


# ============================================================
# Section 4: Text Classification
# ============================================================

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

COUNTRY_NAMES = {
    'af': 'Afghanistan', 'cn': 'China', 'de': 'Germany',
    'fi': 'Finland', 'fr': 'France', 'in': 'India',
    'ir': 'Iran', 'pk': 'Pakistan', 'za': 'South Africa'
}


def train_city_models(train_dir, model_class, n=2, k=1):
    """Train one n-gram model per country.

    Args:
        train_dir: Path to directory containing training files.
        model_class: The model class to use.
        n: N-gram order.
        k: Smoothing parameter.

    Returns:
        A dictionary mapping country codes to trained models.
    """
    models = {}
    for code in COUNTRY_CODES:
        filepath = os.path.join(train_dir, f'{code}.txt')
        if os.path.exists(filepath):
            models[code] = create_ngram_model_lines(model_class, filepath, n=n, k=k)
    return models


def classify_city(city_name, models):
    """Classify a city name by finding the model with lowest perplexity.

    Args:
        city_name: A string (city name to classify).
        models: Dict mapping country codes to trained models.

    Returns:
        The country code (string) of the predicted country.
    """
    best_code = None
    best_pp = float('inf')
    for code, model in models.items():
        pp = model.perplexity(city_name)
        if pp < best_pp:
            best_pp = pp
            best_code = code
    return best_code


def evaluate_classifier(val_dir, models):
    """Compute classification accuracy on the validation set.

    Args:
        val_dir: Path to directory containing validation files.
        models: Dict mapping country codes to trained models.

    Returns:
        Accuracy as a float (correct / total).
    """
    correct = 0
    total = 0
    for code in COUNTRY_CODES:
        filepath = os.path.join(val_dir, f'{code}.txt')
        if not os.path.exists(filepath):
            continue
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            for line in f:
                city = line.strip()
                if not city:
                    continue
                pred = classify_city(city, models)
                total += 1
                if pred == code:
                    correct += 1
    return correct / total if total > 0 else 0.0


# ============================================================
# Main: run all demos and print results
# ============================================================

if __name__ == '__main__':
    SHAKESPEARE_PATH = 'data/shakespeare_input.txt'
    SONNETS_PATH = 'data/shakespeare_sonnets.txt'
    MODERN_ENGLISH_PATH = 'data/modern_english_sample.txt'
    CITIES_TRAIN_DIR = 'data/train'
    CITIES_VAL_DIR = 'data/val'
    CITIES_TEST_PATH = 'data/cities_test.txt'

    # ---- Section 1 sanity checks ----
    print("=== Section 1: N-gram basics ===")
    assert ngrams(0, 'abc') == [('', 'a'), ('', 'b'), ('', 'c')]
    assert ngrams(1, 'abc') == [('~', 'a'), ('a', 'b'), ('b', 'c')]
    assert ngrams(2, 'abc') == [('~~', 'a'), ('~a', 'b'), ('ab', 'c')]
    assert ngrams(1, '') == []
    print("ngrams: OK")

    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    assert m.prob('a', 'b') == 1.0
    assert m.prob('a', 'a') == 0.0
    print("NgramModel prob (no smooth): OK")

    m2 = NgramModel(1, 1)
    m2.update('abab')
    m2.update('abcd')
    assert abs(m2.prob('a', 'a') - 1.0/7.0) < 1e-10
    assert abs(m2.prob('a', 'b') - 4.0/7.0) < 1e-10
    print("NgramModel prob (k=1): OK")

    # ---- Perplexity ----
    print("\n=== Section 2: Perplexity ===")
    assert abs(m.perplexity('abcd') - 1.189207115002721) < 1e-6
    assert m.perplexity('abca') == float('inf')
    print("Perplexity: OK")

    # ---- Shakespeare generation ----
    print("\n=== Shakespeare Text Generation ===")
    for n in [2, 3, 4, 7]:
        model = create_ngram_model(NgramModel, SHAKESPEARE_PATH, n)
        random.seed(42)
        print(f"\nn = {n}:")
        print("=" * 60)
        print(model.random_text(250))

    # ---- Shakespeare perplexity ----
    print("\n=== Perplexity on Shakespeare Sonnets (k=1) ===")
    print(f"{'n':<5} {'Perplexity':<15}")
    print("-" * 20)
    with open(SONNETS_PATH, encoding='utf-8', errors='ignore') as f:
        sonnets_text = f.read()
    for n in [2, 3, 4, 7]:
        model = create_ngram_model(NgramModel, SHAKESPEARE_PATH, n, k=1)
        pp = model.perplexity(sonnets_text)
        print(f"{n:<5} {pp:<15.4f}")

    # ---- Modern English perplexity ----
    print("\n=== Perplexity on Modern English (k=1) ===")
    print(f"{'n':<5} {'Perplexity':<15}")
    print("-" * 20)
    with open(MODERN_ENGLISH_PATH, encoding='utf-8', errors='ignore') as f:
        modern_text = f.read()
    for n in [2, 3, 4, 7]:
        model = create_ngram_model(NgramModel, SHAKESPEARE_PATH, n, k=1)
        pp = model.perplexity(modern_text)
        print(f"{n:<5} {pp:<15.4f}")

    # ---- Interpolation ----
    print("\n=== Interpolation Lambda Analysis (n=3, k=1) ===")
    with open(SONNETS_PATH, encoding='utf-8', errors='ignore') as f:
        sonnets_text = f.read()
    lambda_configs = [
        [0.25, 0.25, 0.25, 0.25],
        [0.6, 0.2, 0.1, 0.1],
        [0.1, 0.1, 0.2, 0.6],
        [0.4, 0.3, 0.2, 0.1],
        [0.5, 0.3, 0.1, 0.1],
    ]
    print(f"{'Lambdas':<38} {'Perplexity':<15}")
    print("-" * 55)
    best_interp_pp = float('inf')
    best_lambdas = None
    for lambdas in lambda_configs:
        model = create_ngram_model(NgramModelWithInterpolation, SHAKESPEARE_PATH, n=3, k=1)
        model.set_lambdas(lambdas)
        pp = model.perplexity(sonnets_text)
        print(f"{str(lambdas):<38} {pp:<15.4f}")
        if pp < best_interp_pp:
            best_interp_pp = pp
            best_lambdas = lambdas
    # non-interpolated baseline
    m_base = create_ngram_model(NgramModel, SHAKESPEARE_PATH, n=3, k=1)
    pp_base = m_base.perplexity(sonnets_text)
    print(f"\nNon-interpolated n=3 (k=1): {pp_base:.4f}")
    print(f"Best interpolated ({best_lambdas}): {best_interp_pp:.4f}")

    # ---- City classification ----
    print("\n=== City Classification Baseline (n=2, k=1) ===")
    models = train_city_models(CITIES_TRAIN_DIR, NgramModel, n=2, k=1)
    acc = evaluate_classifier(CITIES_VAL_DIR, models)
    print(f"Baseline accuracy (n=2, k=1): {acc:.4f}")

    # Hyperparameter search
    print("\n=== Hyperparameter Search ===")
    print(f"{'n':<5} {'k':<8} {'Model':<12} {'Accuracy':<10}")
    print("-" * 40)
    best_acc = 0
    best_config = None
    for n_val in [2, 3, 4, 5]:
        for k_val in [0.1, 0.5, 1, 2]:
            for cls_name, cls in [('NgramModel', NgramModel), ('Interp', NgramModelWithInterpolation)]:
                ms = train_city_models(CITIES_TRAIN_DIR, cls, n=n_val, k=k_val)
                a = evaluate_classifier(CITIES_VAL_DIR, ms)
                print(f"{n_val:<5} {k_val:<8} {cls_name:<12} {a:.4f}")
                if a > best_acc:
                    best_acc = a
                    best_config = (n_val, k_val, cls_name)
    print(f"\nBest config: n={best_config[0]}, k={best_config[1]}, model={best_config[2]}")
    print(f"Best accuracy: {best_acc:.4f}")

    # Error analysis with best model
    best_n, best_k, best_cls_name = best_config
    best_cls = NgramModel if best_cls_name == 'NgramModel' else NgramModelWithInterpolation
    best_models = train_city_models(CITIES_TRAIN_DIR, best_cls, n=best_n, k=best_k)

    print("\n=== Error Analysis ===")
    errors = []
    confusion = defaultdict(Counter)
    total_c = 0
    correct_c = 0
    for code in COUNTRY_CODES:
        filepath = os.path.join(CITIES_VAL_DIR, f'{code}.txt')
        if not os.path.exists(filepath):
            continue
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            for line in f:
                city = line.strip()
                if not city:
                    continue
                pred = classify_city(city, best_models)
                total_c += 1
                confusion[code][pred] += 1
                if pred == code:
                    correct_c += 1
                else:
                    errors.append((city, code, pred))
    print(f"Overall accuracy: {correct_c}/{total_c} = {correct_c/total_c:.4f}")
    print(f"\nFirst 20 misclassified cities:")
    for city, true, pred in errors[:20]:
        print(f"  {city:<25} true={true} ({COUNTRY_NAMES[true]:<15}) pred={pred} ({COUNTRY_NAMES[pred]})")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(f"{'':>4}", end='')
    for c in COUNTRY_CODES:
        print(f"{c:>5}", end='')
    print()
    for true in COUNTRY_CODES:
        print(f"{true:>4}", end='')
        for pred in COUNTRY_CODES:
            print(f"{confusion[true][pred]:>5}", end='')
        print()

    # Write test labels
    best_n2, best_k2, best_cls_name2 = best_config
    best_cls2 = NgramModel if best_cls_name2 == 'NgramModel' else NgramModelWithInterpolation
    best_models2 = train_city_models(CITIES_TRAIN_DIR, best_cls2, n=best_n2, k=best_k2)
    with open(CITIES_TEST_PATH, encoding='utf-8', errors='ignore') as f:
        test_cities = [line.strip() for line in f if line.strip()]
    with open('test_labels.txt', 'w') as f:
        for city in test_cities:
            pred = classify_city(city, best_models2)
            f.write(pred + '\n')
    print(f"\nWrote {len(test_cities)} predictions to test_labels.txt")
