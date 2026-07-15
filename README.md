# Datasheet Extraction

Pull a sensor's specifications out of its datasheet into a typed object, and
measure how often that goes wrong.

The extraction is the easy half. Claude's structured outputs enforce the schema
server-side, so a response either validates against `SensorSpec` or the call
fails — there is no parse-the-model's-JSON step to go wrong. The half worth
building is the evaluation: knowing, per field, how often the answer is right,
how often it's missing, and **how often the model invents a specification the
document never stated.**

That last number is the one this repo exists for.

## Why hallucination rate, specifically

A datasheet is mostly *absence*. A product brief states five things and stays
silent about eleven others, and "not stated" is the correct answer for those
eleven — not a gap in the data, but a fact about the document that a model can
get right or wrong.

Precision and recall computed only over stated fields cannot see this. A model
that reads the `TRX-9` brief, correctly reports its five specs, and then fills
in a plausible operating temperature range that appears nowhere in the text
scores **precision 1.0** — and is the model you must never put in front of a
procurement decision, because a fabricated spec propagates silently while a null
is visibly missing.

So every field on every document lands in one of five buckets:

| | gold has a value | gold is null |
| --- | --- | --- |
| **model returned a value** | match → `TRUE_POSITIVE`<br>mismatch → `WRONG_VALUE` | `HALLUCINATION` |
| **model returned null** | `FALSE_NEGATIVE` | `TRUE_NEGATIVE` |

`hallucination_rate` is scored over the right-hand column only: the chances the
model had to invent something, and how often it took them.

`WRONG_VALUE` counts as **both** a false positive and a false negative. The model
both failed to produce the right answer and asserted a wrong one, and a metric
that charged it once would rank "confidently wrong" level with "said nothing" —
which is backwards for anything downstream of the extraction.

## Running it

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...          # or: ant auth login

python -m datasheet_extraction evaluate --corpus corpus/demo
```

That prints a row per field:

```
field                 support  precision  recall  f1      hallucination_rate
--------------------  -------  ---------  ------  ------  ------------------
part_number           4        ...
center_frequency_ghz  2        ...
...
ALL                   40       ...
```

**There are no measured numbers in this README.** Filling that table in requires
running the thing against the API, and the numbers it produces on `corpus/demo`
would say nothing anyway — four documents hand-written to match their own labels
measure the harness, not a model. Real numbers need a real corpus; see
[corpus/README.md](corpus/README.md).

## The ablation

The question is never "is Opus better than Haiku" — it is whether it's better by
enough to justify five times the input price on *this* task. So the sweep reports
accuracy next to what the accuracy cost:

```bash
python -m datasheet_extraction ablate \
  --models claude-opus-4-8 claude-sonnet-5 claude-haiku-4-5 \
  --variants minimal strict strict_with_examples \
  --output ablation.json
```

```
model  variant  f1  precision  recall  halluc_rate  cost_usd  latency_s  failed
```

Two axes, because both are hypotheses worth testing rather than assuming:

- **model** — the cost/quality frontier.
- **prompt variant** — `minimal` leans entirely on the schema's field
  descriptions; `strict` adds explicit abstention and unit rules;
  `strict_with_examples` adds worked cases for the two conversions most likely
  to go wrong (half-angle fields of view, frequency bands). Whether the extra
  prompt tokens pay for themselves is measurable, so it should be measured.

## Layout

```
src/datasheet_extraction/
  schema.py     SensorSpec — every field nullable, because "not stated" is an answer
  prompts.py    prompt variants, as ablation arms
  extract.py    the API calls: structured outputs, usage capture, failure capture
  compare.py    is this field right? numeric tolerance, text normalisation
  evaluate.py   outcomes -> precision / recall / F1 / hallucination rate
  cost.py       token accounting and USD estimates
  corpus.py     loading documents + gold labels, and refusing inconsistent ones
  cli.py        extract / evaluate / ablate
corpus/demo/    four synthetic datasheets and their labels
tests/          64 tests, no API key required
```

## Design notes

**Failures are data, not exceptions.** A corpus run must not lose forty good
extractions because the forty-first hit a rate limit, so API errors, refusals,
and truncated responses are captured per document. The evaluator scores a failed
extraction as a miss on every field rather than skipping it — an extraction that
crashed is a failure, not an absence of evidence.

**Numeric comparison is relative, not exact.** `77` and `77.0` are the same
answer; 1% tolerance absorbs rounding without absorbing real errors. Absolute
tolerance would be nonsense across fields spanning `0.04 m` to `4000 MHz`.

**Prompt caching is wired up and off by default, because here it would do
nothing.** The cacheable prefix is the system prompt, and every variant in this
repo is a few hundred tokens — far below the 1024–4096 token minimum a prefix
needs before any model will cache it. Marking a shorter prefix isn't an error;
it's *silently* inert, which is worse. `--cache-system` only attaches
`cache_control` when the prompt could plausibly clear the threshold for that
model, and `cache_read_tokens` in the run summary is there to check it rather
than trust it. Caching starts paying when the instructions grow — a long
extraction rulebook, few-shot examples, a fuller schema.

**Batch API for bulk runs.** `submit_batch` halves the price for work that
doesn't need to be interactive, which an ablation sweep doesn't. Results come
back in arbitrary order and are keyed by `custom_id`, never by position.

**Tests run without an API key.** The client is faked, so CI proves the
extraction and scoring logic without spending anything or depending on the API
being reachable. The metric code is the part most worth testing and the part
least in need of a network.

## Known limits

- **Text in, text out.** The loader takes plain text; getting there from a PDF is
  out of scope. That choice probably costs more accuracy than any prompt variant
  here — a spec table flattened into prose loses the column headers that say
  which number is typical and which is maximum — and it deserves its own ablation.
- **Single-label ground truth.** Each field has one right answer. Datasheets that
  give a spec per operating mode don't fit the schema.
- **String matching is normalised-exact.** "Texas Instruments" and "Texas
  Instruments Inc." are scored as different answers. Deliberate — a fuzzy
  comparator would quietly inflate the numbers — but it means `manufacturer`
  precision is a stricter measure than it looks.
- **Prices are hardcoded** as of 2026-06-24 and will drift.
