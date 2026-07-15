# Datasheet Extraction

[![ci](https://github.com/Bilalovski/datasheet_extraction/actions/workflows/ci.yml/badge.svg)](https://github.com/Bilalovski/datasheet_extraction/actions/workflows/ci.yml)

Pull a sensor's specifications out of its datasheet into a typed object, and
measure how often that goes wrong. Runs against the [DeepSeek](https://api-docs.deepseek.com)
API.

The extraction is the easy half. The half worth building is the evaluation:
knowing, per field, how often the answer is right, how often it's missing, and
**how often the model invents a specification the document never stated.**

That last number is the one this repo exists for.

## Why hallucination rate, specifically

A datasheet is mostly *absence*. A product brief states five things and stays
silent about eleven, and "not stated" is the correct answer for those eleven —
not a gap in the data, but a fact about the document a model can get right or
wrong.

Precision and recall computed only over stated fields cannot see this. A model
that reads the `TRX-9` brief, correctly reports its five specs, and then fills in
a plausible operating temperature range that appears nowhere in the text scores
**precision 1.0** — and is the model you must never put in front of a procurement
decision, because a fabricated spec propagates silently while a null is visibly
missing.

So every field of every document lands in one of five buckets:

| | gold has a value | gold is null |
| --- | --- | --- |
| **model returned a value** | match → `TRUE_POSITIVE`<br>mismatch → `WRONG_VALUE` | `HALLUCINATION` |
| **model returned null** | `FALSE_NEGATIVE` | `TRUE_NEGATIVE` |

`hallucination_rate` is scored over the right-hand column only: the chances the
model had to invent something, and how often it took them.

`WRONG_VALUE` counts as **both** a false positive and a false negative. The model
both failed to produce the right answer and asserted a wrong one, and a metric
that charged it once would rank "confidently wrong" level with "said nothing" —
backwards for anything downstream.

## Results, and what they don't show

Measured 2026-07-16 on `corpus/demo`, 4 documents × 16 fields, 41 stated fields
and 23 unstated. Raw output: [`results/`](results/).

| model | variant | F1 | halluc. rate | cost (USD) | latency | repairs |
| --- | --- | --- | --- | --- | --- | --- |
| v4-flash | minimal | 1.0 | 0.0 | $0.00102 | 3.7 s | 0 |
| v4-flash | strict | 1.0 | 0.0 | $0.00119 | 4.5 s | 0 |
| v4-flash | strict_with_examples | 1.0 | 0.0 | $0.00064 | 4.0 s | 1 |
| v4-pro | minimal | 1.0 | 0.0 | $0.00376 | 6.8 s | 0 |
| v4-pro | strict | 1.0 | 0.0 | $0.00344 | 5.7 s | 0 |
| v4-pro | strict_with_examples | 1.0 | 0.0 | $0.00377 | 6.4 s | 0 |

**Every configuration scores 1.0. The demo corpus is saturated.** Six
configurations, one distinct F1 value between them — it cannot tell any model or
prompt apart, so none of these rows is evidence that any model is good at this
task. It is evidence that four clean, unambiguous documents written to match
their own labels are easy, which was true before the run and is why the corpus
ships as a fixture rather than a benchmark. Real datasheets — PDF-mangled tables,
footnoted conditions, per-mode specs — are where the metric would start to
separate things. See [corpus/README.md](corpus/README.md).

Two things the sweep *does* measure honestly:

- **v4-pro costs 3.8× more and runs 1.6× slower, for identical accuracy here.**
  On a saturated corpus that's the expected shape of the result, and it's the
  question the ablation exists to answer on a real one.
- **One schema repair fired.** DeepSeek does not enforce the schema server-side,
  and the model returned a null-ish string where a number-or-null belonged. Not
  hypothetical — see below.

The whole sweep cost **$0.0138**.

## Running it

```bash
pip install -e ".[dev]"
export DEEPSEEK_API_KEY=...        # never hardcode this; see Credentials

python -m datasheet_extraction evaluate --corpus corpus/demo
python -m datasheet_extraction ablate --output ablation.json
```

Models are `deepseek-v4-flash` (default) and `deepseek-v4-pro`.

> **`deepseek-chat` and `deepseek-reasoner` retire on 2026-07-24.** They are
> aliases for v4-flash in non-thinking and thinking mode. This repo defaults to
> the canonical ids; the aliases still price correctly so old runs don't crash.

### Credentials

The key is read from `DEEPSEEK_API_KEY` and **has no fallback default**, which is
deliberate. A hardcoded key in a public repo is scraped from GitHub's push
firehose within minutes, and deleting the line later doesn't remove it from git
history. `tests/test_extract.py` asserts no `sk-` string exists in `src/`.

## What DeepSeek's API forced, and how it was found out

Both of these came from probing the API, not from assuming it behaves like
OpenAI's:

**There is no strict schema mode.** `response_format={"type": "json_schema"}` is
rejected outright — *"This response_format type is unavailable now"*. So the
schema travels as a **function's parameters** and is enforced client-side by
Pydantic. This also happens to be the better channel: passing `SensorSpec`'s JSON
schema as the tool parameters is what carries the field descriptions — the unit
rules, the half-angle convention, "null if not stated" — to the model. The prompt
engineering lives in the type.

Client-side validation is a real step, not a formality. `deepseek-reasoner`
returns `"elevation_fov_deg": "null"` — a *quoted string* where the schema says
number-or-null. A null-ish string is repaired to `None` and **counted**, so the
extraction metric measures extraction rather than JSON etiquette while the
serialisation defect stays visible in the `repairs` column. A genuine type error
(`"quite far"` in a numeric field) is a failure, not a repair.

**Forced `tool_choice` only works in non-thinking mode.** Every canonical model id
runs in thinking mode and rejects `tool_choice="required"` or a named function
with *"Thinking mode does not support this tool_choice"*. `tool_choice="auto"` is
the only mechanism that works across all of them, so that's what's used — and a
turn where the model answers in prose instead of calling the tool is a recorded
failure, because with `auto` that's reachable.

| model id | resolves to | forced `tool_choice` | `auto` | `json_object` |
| --- | --- | --- | --- | --- |
| `deepseek-chat` | v4-flash, non-thinking | ✅ | ✅ | ✅ |
| `deepseek-reasoner` | v4-flash, thinking | ❌ | ✅ | ✅ |
| `deepseek-v4-flash` | v4-flash, thinking | ❌ | ✅ | ✅ |
| `deepseek-v4-pro` | v4-pro, thinking | ❌ | ✅ | ✅ |

## Layout

```
src/datasheet_extraction/
  schema.py     SensorSpec — every field nullable, because "not stated" is an answer
  prompts.py    prompt variants, as ablation arms
  extract.py    the API calls: tool-schema delivery, validation, failure capture
  compare.py    is this field right? numeric tolerance, text normalisation
  evaluate.py   outcomes -> precision / recall / F1 / hallucination rate
  cost.py       token accounting against DeepSeek's published rates
  corpus.py     loading documents + gold labels, and refusing inconsistent ones
  cli.py        extract / evaluate / ablate
corpus/demo/    four synthetic datasheets and their labels
results/        raw ablation output
tests/          76 tests, no API key required
```

## Design notes

**Failures are data, not exceptions.** A corpus run must not lose forty good
extractions because the forty-first hit a rate limit, so API errors, prose
answers, malformed JSON, and schema violations are captured per document. The
evaluator scores a failed extraction as a miss on every field rather than
skipping it — an extraction that crashed is a failure, not an absence of
evidence.

**Numeric comparison is relative, not exact.** `77` and `77.0` are the same
answer; 1% tolerance absorbs rounding without absorbing real errors. Absolute
tolerance would be nonsense across fields spanning `0.04 m` to `4000 MHz`.

**Caching needs no code.** DeepSeek caches context automatically and reports the
hit/miss split per request — no `cache_control`, no minimum-prefix rules, nothing
to opt into. A cache hit costs **~2% of a miss** ($0.0028 vs $0.14 per MTok on
v4-flash), which is a steeper discount than most providers give. The shared
system prompt across a corpus run measured a **44% hit rate** on the four-document
demo, and that rate rises with corpus size. `cost.py` bills hits and misses at
their real separate rates instead of averaging them.

**Tests run without an API key.** The client is faked, so CI proves the extraction
and scoring logic without spending anything or depending on the API being up. The
faked failures are the ones the real API actually produces — quoted `"null"`,
prose instead of a tool call — because those were observed first.

## Known limits

- **Text in, text out.** The loader takes plain text; getting there from a PDF is
  out of scope, and that choice probably costs more accuracy than any prompt
  variant here. It deserves its own ablation.
- **Single-label ground truth.** Each field has one right answer. Datasheets
  giving a spec per operating mode don't fit the schema.
- **String matching is normalised-exact.** "Texas Instruments" and "Texas
  Instruments Inc." score as different answers. Deliberate — a fuzzy comparator
  would quietly inflate the numbers — but `manufacturer` precision is stricter
  than it looks.
- **Prices are hardcoded** as read on 2026-07-15 and will drift.
