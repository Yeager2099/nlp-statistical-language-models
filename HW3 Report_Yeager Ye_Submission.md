# Report transcription

Text extracted from the author-provided course report. Figures are not reproduced and tables may lose formatting. Student identifiers and institutional email addresses are removed.

## Page 1

CIS 5300: Natural Language Processing
Homework 3: Probability and Language Models
Report Template
Name: Yeager Ye (QIUYU YE)
Penn ID: [STUDENT_ID_REDACTED]
Learning Objectives
1. Compute n-gram probabilities using maximum likelihood estimation.
2. Implement smoothing and interpolation to handle unseen n-grams.
3. Generate text by sampling from learned distributions.
4. Evaluate language models using perplexity.
5. Apply language models to classification tasks using per-class perplexity.
Points Breakdown
Section Points Description
Answer 2.2 5 pts Text Generation Quality Across N Values
Answer 2.3 5 pts Perplexity Analysis
Answer 3.2 5 pts Interpolation Lambda Analysis
Answer 4.3 5 pts City Classification Error Analysis

## Page 2

CIS 5300: Natural Language Processing
Answer 5.1 5 pts The Markov Assumption
Answer 5.2 5 pts Smoothing and Zero Probabilities
Answer 5.3 5 pts N-gram Models vs. Neural Language Models
Total 35 pts
Answer 2.2: Text Generation Quality Across N Values
Analyze how the quality of Shakespeare-style generated text changes as you vary
n (n=2, 3, 4, 7).
● Record the generated text for each value of n.
n Generated Text
Fir sor if make dot
mor whalmothey my the de dame hantiong huselphe ginst resce nothe you ang amustraten seartune
woreceiror
O vou,
2
Will to mink for fuld ace.
Butly eve wisen and sove hon lif whou youst man huntemstan'd; hich, at con as not.
SPUCES:
First the did recon the what, they must said fair elevil they motince crue.
Reft for curb and May your let live may'd of my Am suppert neven!
SILVIA:
3
O, and deat woman at know me, puledge me from Itall.
CLAUS:
I writ horsh not wantager on off them
4 First see

## Page 3

CIS 5300: Natural Language Processing
n Generated Text
Lie speak
it, what other's not a fain silent; and your read cut of: I never you art abysm
of men into payment to makes; best master, that cause again,
Return,
As is banishethere been whom zeal:
Knocks removed horse please the the heel, and
First Citizen:
Peace, for I will wear the elder of her mask'd.
The Earl of March.
PRINCE EDWARD:
What man indeed to him: that's that?
7
BRUTUS:
I loved it to hear what a word:
And thus for policy.
SIR TOBY BELCH:
Would you both the peasant slumber.
● Compare the generations. Does using a longer context produce more
coherent text?
Yes. At n=2, the output is almost pure character-level noise with only accidental English
fragments. At n=3, recognizable words and Shakespearean character names begin to
emerge. At n=4, plausible short phrases and punctuation patterns appear, though they
are still joined incoherently. At n=7, the model produces complete, grammatical
sentences with correct stage directions and character names closely mimicking real
Shakespeare. Longer context consistently produces more coherent text.
● What kinds of patterns appear at each value of n?
n=2: Random character sequences; occasional vowel-consonant alternation but no real
words. n=3: Recognizable English words ('said', 'must', 'not') and stage-name fragments
('SILVIA:', 'CLAUS:'), though sentences lack grammatical structure. n=4: Short meaningful
phrases ('not a fain silent', 'best master'), correct punctuation placement, and partial line
breaks. n=7: Full proper sentences, correct speaker labels, dramatic dialogue structure,
and coherent clause boundaries essentially verbatim substrings from the training corpus.

## Page 4

CIS 5300: Natural Language Processing
● Connect your observations to the Markov assumption (Module 3: Probabilistic
Language Models lecture). Why does a longer context help? What is the
model losing when n=1 that it gains when n=7?
The Markov assumption states that the probability of each character depends only on
the n-1 preceding characters. When n=1 (unigram), the model conditions on nothing:
each character is sampled purely from its marginal frequency, so the output is
structurally random. As n increases, the model conditions on a longer prefix, capturing
word-level patterns (n=3), phrase-level patterns (n=4), and near-complete syntactic
units (n=7). What the model gains by going from n=1 to n=7 is the ability to enforce
character-sequence constraints that encode word boundaries, morphology, punctuation
conventions, and dialogue structure all of which require at least a few characters of
context to be predictable.
● What's the catch? What goes wrong (or what would go wrong on a small
training corpus) as n keeps increasing?
As n grows, the number of distinct n-gram contexts grows exponentially (O(|V|^n)), while
the training corpus size remains fixed. Most high-order contexts are observed only once
or not at all, making maximum-likelihood estimates unreliable. At n=7 on the full
Shakespeare corpus (~4.5 MB), the model essentially memorizes verbatim substrings.
On a small training corpus ‘say, a single sonnet’ a 7-gram model would have near-zero
probability for almost every context in a test text, causing generation to stall or perplexity
to explode. The model degrades from learning a distribution to pure memorization, with
no generalization ability.
● Connect your answer to data sparsity and the role of smoothing.
Data sparsity is precisely the consequence of having exponentially many possible
contexts but only a finite number of observations. Without smoothing (k=0), any n-gram
context unseen in training receives probability zero, causing perplexity to be infinite on
any realistic test set. Add-k smoothing assigns a small non-zero probability to unseen
n-grams by adding k to every count, redistributing a fraction of the probability mass from
seen events to unseen ones. However, smoothing is only a partial remedy: as n
increases, the fraction of unseen contexts grows so large that even a smoothed model's
probabilities become nearly uniform, losing all discriminative power. This is why
perplexity rises sharply at n=7 despite k=1 smoothing.

## Page 5

CIS 5300: Natural Language Processing
Answer 2.3: Perplexity Analysis
Report and interpret the perplexity values you computed for different n-gram
models.
● Record the perplexity values that you get for n=2, 3, 4, and 7 on the
Shakespeare sonnets text.
n Perplexity
2 7.9101
3 6.1222
4 6.4856
7 21.6897
● Which n-gram order gives the lowest perplexity? Why?
n=3 gives the lowest perplexity (6.1222). Trigram contexts are long enough to capture
the character-level word structure and stylistic patterns shared between Shakespeare's
plays (training) and sonnets (test), while being short enough that most trigrams in the
sonnets were also seen during training. n=4 is slightly worse (6.4856) because higher-
order contexts are less likely to match exactly, even between related Shakespeare
texts, increasing reliance on smoothed probabilities. n=7 is dramatically worse
(21.6897) due to extreme sparsity.
● What does it mean when perplexity is infinity in practical terms? When does
this occur?
Perplexity is computed as exp(--(1/N) * sum of log P(c_i | context)). If any single P(c_i |
context) = 0, then log(0) = -infinity, making the entire sum -infinity and therefore
perplexity = +infinity. In practical terms, an infinite perplexity means the model assigns
zero probability to at least one character in the test text, i.e., that sequence is completely
impossible under the model. This occurs whenever k=0 (no smoothing) and a character
follows a context never observed in training.
● Compute the perplexity of the Shakespeare play model on the modern
English Wikipedia excerpt. Record the perplexity values for n=2, 3, 4, and 7.

## Page 6

CIS 5300: Natural Language Processing
n Perplexity
2 13.8909
3 11.7169
4 12.6790
7 34.7093
● Compare the perplexity of Shakespeare sonnets vs. the modern English
Wikipedia excerpt. What does the difference tell you about the model?
Across all n values, perplexity on modern English Wikipedia is roughly 1.7x to 1.6x
higher than on the Shakespeare sonnets. For n=3 the gap is 11.7169 vs. 6.1222. This
tells us that the model is domain-specific: it was trained on Early Modern English with
archaic vocabulary, poetic inversions, and stylized punctuation. Modern Wikipedia
prose uses different character sequences (contemporary spellings, numerals, place
names, technical terms) that were rarely or never seen in the training corpus. The higher
perplexity quantifies exactly how much more surprised the model is when encountering
out-of-domain text (i.e., how poorly it generalizes across writing styles.)
Answer 3.2: Interpolation Lambda Analysis
Analyze the effect of different lambda weight configurations on interpolated
model performance.
● List at least 4 different lambda settings and compute the resulting perplexity.
Lambda Settings Perplexity
[0.25, 0.25, 0.25, 0.25] 8.1888
[0.6, 0.2, 0.1, 0.1] 6.6346
[0.1, 0.1, 0.2, 0.6] 11.7181
[0.4, 0.3, 0.2, 0.1] 7.0724
[0.5, 0.3, 0.1, 0.1] 6.7588

## Page 7

CIS 5300: Natural Language Processing
● Which configuration gave the lowest perplexity?
The configuration [0.6, 0.2, 0.1, 0.1] gave the lowest interpolated perplexity of 6.6346.
This places the highest weight (0.6) on the highest-order component (n=3) and
diminishing weights on lower-order components, matching the empirical finding that
trigram context is most predictive for this corpus.
● What do the relative weights tell you about the importance of different
context lengths for this task? For example, if the n=3weight is highest, what
does that mean?
A high weight on the n=3 component means that trigram context (the three most
recent characters) carries the majority of predictive power for Shakespeare-style
character sequences. Lower weights on bigram (n=2) and unigram (n=1) components
indicate that shorter contexts add relatively little signal on top of the trigram. In
contrast, the worst-performing configuration [0.1, 0.1, 0.2, 0.6] places the most weight
on the unigram, which knows only character marginal frequencies and ignores all
sequential context. Hence, the high perplexity of 11.7181, nearly matching the
unigram perplexity.
● Compare the best interpolated model's perplexity to the non-interpolated
n=3 model with the same k. Report the values below:
Interpolated Model (n=3) |
Uninterpolated Model (n=3)
Best lambda = [0.6, 0.2, 0.1, 0.1]
Perplexity = 6.6346 Perplexity = 6.1222
● Does interpolation help? Why or why not?
On in-domain text (sonnets evaluated against a Shakespeare-trained model),
interpolation does NOT help: the best interpolated perplexity (6.6346) is higher than the
non-interpolated n=3 perplexity (6.1222). When the training and test text are stylistically
similar and the corpus is large enough, the trigram model alone is already well-
estimated, and mixing in lower-order components introduces noise. Interpolation is thus
most beneficial when the test domain differs from training.
● Consider also the case of using your model against out-of-domain text.

## Page 8

CIS 5300: Natural Language Processing
When the Shakespeare-trained model is applied to modern English Wikipedia, the non-
interpolated n=3 model has perplexity 11.7169. An interpolated model with even
weighting [0.25,0.25,0.25,0.25] yields approximately 12.7 in this scenario, still slightly
worse here, but the key advantage of interpolation would emerge more strongly with a
more extreme domain shift or a smaller training corpus. In those settings, high-order
contexts become almost entirely unseen, and the model must fall back on the unigram or
bigram components to avoid near-infinite perplexity. Interpolation thus acts as a graceful
degradation mechanism: even when the n=3 context is unseen, the model can still
assign a reasonable probability using lower-order statistics.
Answer 4.3: City Classification Error Analysis
Report your best configuration and its validation accuracy. (From Problem 4.2)
Parameter Value
N-gram order 4
Smoothing parameter (k) 0.1
Model class NgramModelWithInterpolation
Validation accuracy 79.46% (1230 / 1548)
Analyze the errors your language-model-based classifier makes on the city name
classification task.
● Which country pairs are most commonly confused? Why might this be the
case? Consider the confusion matrix.What linguistic patterns (character
sequences, suffixes, character distributions) might explain these confusions?
The most common confusion pairs visible in the matrix are India (IN) confused with
Pakistan (PK): 29 Indian cities predicted as Pakistani. France (FR) confused with
Germany (DE): 17 French cities predicted as German. Iran (IR) confused with India (IN):
11 cases. Afghanistan (AF) confused with Iran (IR): 6 cases. India-Pakistan confusion
reflects shared South Asian naming roots: both Hindi and Urdu city names heavily use '-
abad', '-pur', '-nagar', '-bad' suffixes, which are virtually identical in romanized form.
France-Germany confusion arises from shared Germanic and Latin-derived roots: many
French cities near the Rhine border (Alsace, Lorraine) have German-sounding names
(Strasbourg, Colmar), and both languages use '-en', '-er', '-burg', '-dorf' patterns. Iran-

## Page 9

CIS 5300: Natural Language Processing
Afghanistan confusion stems from the shared Dari/Persian phonological space: both use
'shahr', 'khan', 'abad', 'kand' patterns in city names.
Confusion Matrix (rows = true label, columns = predicted label; green cells = correct predictions):
True \ Pred AF CN DE FI FR IN IR PK ZA
AF 16 1 1 0 0 2 6 5 0
CN 4 270 3 3 2 9 2 1 6
DE 1 3 268 7 5 4 4 1 7
FI 0 0 1 75 1 5 0 1 0
FR 1 1 17 10 252 3 2 3 11
IN 11 3 2 10 7 220 9 29 9
IR 8 1 1 3 2 11 51 5 2
PK 9 2 1 2 1 17 2 16 4
ZA 3 1 11 4 9 3 2 1 62
● Provide at least 3 specific examples of misclassified cities and explain why
the model got them wrong. Explain what n-gram patterns might cause the
confusion.
(1) shahran (true=AF, pred=IR): Contains the prefix 'shahr-' (meaning 'city' in
Persian/Dari), which is extremely common in Iranian city names. The 4-gram contexts
'shah', 'ahri', 'hran' all have high probability under the Iran model because Iran has
many 'shahr-' prefixed cities in training.
(2) narang (true=AF, pred=IN): The suffix '-ang' is a common ending in Indian city
names (e.g., Warangal, Kohlapur). The trigrams 'ran', 'ang' are strongly associated
with India in the trained model.
(3) koktokay (true=CN, pred=FI): This city in Xinjiang (China) has a Kazakh-origin
name. Its character sequences 'ok', 'ko', 'tok', 'ay' resemble Finnish city names with
their characteristic double-consonant and '-kay' patterns, leading the Finland model to
assign it lower perplexity than the China model.
● What linguistic patterns distinguish city names from different countries? For
example, what character sequences are distinctive of Finnish vs. French city
names?
Finnish city names frequently end in '-la', '-lae', '-ki', '-nen', '-jaervi' and contain long
vowel sequences (aa, oo, uu, ae, oe) that are diagnostic. French names often include
'eau', '-sur-', '-en-', '-les-', and nasal vowel approximations like 'ain', 'oin'. German
names favor '-berg', '-burg', '-dorf', '-furt', '-heim'. Pakistani and Indian names share '-

## Page 10

CIS 5300: Natural Language Processing
abad', '-pur', '-nagar', '-bad'. South African names carry Zulu/Xhosa-origin patterns
with 'umh-', 'nko-', 'mbom-', 'nta-'. The n-gram classifier successfully exploits these
character-level distributional signatures but fails when names straddle multiple cultural
or historical influences.
Answer 5.1: The Markov Assumption
● Why is the Markov assumption necessary? What would happen if we tried to
condition on the full history instead? (Think about parameter estimation and
data sparsity.)
The Markov assumption is necessary because conditioning on the full history is
statistically intractable. The full chain rule requires estimating P(w_t | w_1, w_2, ..., w_{t-
1}) for every possible sequence prefix. The number of unique prefixes grows
exponentially with length: even a 50-character sequence has at most a handful of
occurrences in any realistic corpus. Maximum-likelihood estimation becomes
meaningless when nearly every context is seen at most once. The Markov assumption
limits the conditioning window to the most recent n-1 tokens, reducing the parameter
space from O(|V|^T) to O(|V|^n), which is estimable from finite data.
● What long-range dependencies does it miss? Give a concrete example of a
sentence (or word) where a bigram model would fail to capture an important
dependency that a human would recognize.
The Markov assumption misses all dependencies beyond the context window. A concrete
example: in the sentence 'The government, after years of debate and heated
parliamentary sessions, finally passed the bill,' a bigram model sees 'passed' preceded
only by 'finally', completely losing the subject 'government' that determines the semantic
plausibility of the verb.
As a result, the model treats 'the cat passed the bill' and 'the government passed the bill'
as equally likely given 'finally passed'. Similarly, subject-verb agreement—'The keys to
the cabinet ARE on the table' vs. 'The key to the cabinet IS on the table'—requires
connecting 'are/is' back to 'keys/key' across a long intervening clause, which is
impossible for a short n-gram window.

## Page 11

CIS 5300: Natural Language Processing
● How does the Markov assumption simplify the chain rule of probability?
The full chain rule states: P(w_1, w_2, ..., w_n) = prod_i P(w_i | w_1, ..., w_{i-1}).
Under the n-gram Markov assumption, each conditional factor reduces to P(w_i | w_{i-
n+1}, ..., w_{i-1})--only the last n-1 tokens matter. This yields the tractable
approximation: P(w_1, ..., w_n) approx prod_i P(w_i | w_{i-n+1}, ..., w_{i-1}). The
exponentially parameterized joint distribution is converted into a product of simple
conditional distributions, each estimated from a fixed-width sliding window. Both
parameter estimation and probability computation become linear in sequence length.
Answer 5.2: Smoothing and Zero Probabilities
● Why is zero probability a problem for perplexity? What specific mathematical
issue arises when computing perplexity and one of the conditional
probabilities is 0?
Perplexity is defined as PP = exp(--(1/N) * sum_i log P(c_i | context)). When any
single conditional probability P(c_i | context) = 0, log(0) = -infinity. This makes the
entire sum equal to -infinity, and therefore PP = exp(-(-infinity)) = +infinity. The
mathematical issue is that the log-probability is undefined at zero, so one unseen n-
gram is enough to make the perplexity score completely meaningless, even if every
other character in the test set is perfectly predicted. Any realistic test set will contain
some n-gram context not seen in training, making zero probability an inevitable
problem without smoothing.
● How does add-k smoothing fix this problem? Describe the mechanism. What
happens to the probability mass - where does it come from?
Add-k smoothing replaces the raw count with (count + k) before normalizing:
P_smooth(c | context) = (count(context, c) + k) / (count(context) + k * |V|). For any
character c never observed after a given context, count(context, c) = 0, but the
numerator becomes k > 0, guaranteeing a strictly positive probability. The
denominator increases by k * |V|, accounting for the mass added to all |V| possible
characters. The extra probability mass 'comes from' the observed events: each seen
event's probability is reduced slightly to fund the non-zero probabilities of unseen
events. This redistribution ensures no probability is exactly zero, eliminating infinite
perplexity.
● What happens if k is set too large (e.g., k = 1000 or k = 1000000)? How does
this affect the model's probability distribution? Why is this bad?

## Page 12

CIS 5300: Natural Language Processing
When k is very large, k * |V| dominates the denominator regardless of the actual
context count, so P(c | context) = (count + k) / (count + k*|V|) approaches k / (k*|V|) =
1/|V| for every context and character. The model degenerates to a uniform distribution
over the entire vocabulary. All learned n-gram statistics are effectively discarded;
every character becomes equally likely regardless of context. This is bad for two
reasons: (1) Perplexity approaches log(|V|), the maximum possible value, meaning
the model is maximally uncertain. (2) Any downstream task, like text generation,
classification, perplexity evaluation that depends on the model's learned preferences
fails completely, because the model no longer encodes any information about the
training corpus.
Answer 5.3: N-gram Models vs. Neural Language Models
● Identify at least two fundamental limitations of n-gram language models.
Explain why it is a problem for n-gram models
● For each limitation, explain how neural language models (e.g., RNNs,
Transformers) overcome it.
Limitation How neural language models (e.g.,
RNNs, Transformers) overcome it.
Fixed context window. N-gram models RNNs and LSTMs maintain a hidden state
condition only on the n-1 preceding tokens. vector that accumulates information across
Any dependency beyond this window is the entire input sequence, theoretically
completely invisible. For example, in 'The allowing dependencies of arbitrary length.
keys to the cabinet are on the table,' a Transformer models use self-attention, which
trigram model cannot connect 'are' back to directly computes weighted relationships
'keys' for subject-verb agreement because between every pair of positions in the input,
the noun is many tokens earlier. All long- making the effective context window the
range semantic coherence, coreference, entire sequence. Neither architecture
and discourse structure are lost. Increasing enforces a hard context cutoff, so both can in
n only partially helps and rapidly causes principle capture long-range dependencies
the data sparsity problem described above. that n-gram models structurally cannot.
No semantic generalization (lack of Neural language models learn dense vector
distributed representations). N-gram embeddings (word or character embeddings)
models treat each token as an atomic, that place semantically and syntactically
unrelated symbol. 'cat' and 'dog' are similar tokens near each other in a
treated as completely distinct entities with continuous vector space. Because 'cat' and
no shared information, even though they 'dog' end up with similar embeddings, the

## Page 13

CIS 5300: Natural Language Processing
are semantically and syntactically model's learned weights generalize: patterns
interchangeable in most contexts. If 'the observed for 'cat' automatically transfer to
cat sat on the mat' was seen in training but 'dog' in similar contexts. This distributed
'the dog sat on the mat' was not, the n- representation allows neural models to
gram model assigns 'dog' no higher generalize far better to unseen word-context
probability than any other unseen word in combinations, dramatically reducing the
that context, wasting all knowledge about effective sparsity problem that cripples high-
word similarity. This severely limits sample order n-gram models and enabling the model
efficiency: every word-context pair must be to leverage world knowledge encoded across
observed independently. the entire vocabulary.
