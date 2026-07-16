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

A datasheet is mostly *absence*. A sensor states a dozen things and stays silent
about a dozen more, and "not stated" is the correct answer for the silent ones —
not a gap in the data, but a fact about the document a model can get right or
wrong.

Precision and recall computed only over stated fields can't see this. A model
that reads a temperature-sensor datasheet, correctly reports its specs, and then
fills in a plausible field-of-view that appears nowhere in the text scores
**precision 1.0** — and is the model you must never trust, because a fabricated
spec propagates silently while a null is visibly missing.

So every field of every document lands in one of five buckets:

| | gold has a value | gold is null |
| --- | --- | --- |
| **model returned a value** | match → `TRUE_POSITIVE`<br>mismatch → `WRONG_VALUE` | `HALLUCINATION` |
| **model returned null** | `FALSE_NEGATIVE` | `TRUE_NEGATIVE` |

`hallucination_rate` is scored over the right-hand column only: the chances the
model had to invent something, and how often it took them.

`WRONG_VALUE` counts as **both** a false positive and a false negative — the model
both failed to produce the right answer and asserted a wrong one, and charging it
once would rank "confidently wrong" level with "said nothing."

## Results

| model | variant | F1 | precision | recall | halluc. rate | cost | latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v4-flash | minimal | 0.758 | 0.753 | 0.763 | 0.194 | $0.027 | 19 s |
| v4-flash | **strict** | **0.809** | 0.753 | 0.875 | **0.181** | $0.027 | 18 s |
| v4-flash | strict+examples | 0.770 | 0.713 | 0.838 | 0.194 | $0.027 | 15 s |
| v4-pro | minimal | 0.793 | 0.734 | 0.863 | 0.208 | $0.084 | 28 s |
| v4-pro | strict | 0.811 | 0.747 | 0.888 | 0.208 | $0.084 | 27 s |
| v4-pro | strict+examples | 0.796 | 0.729 | 0.875 | 0.222 | $0.087 | 32 s |

*Measured 2026-07-17 on `corpus/sensors` — 8 real datasheets, hand-labelled. Raw
output in [`results/`](results/); reproduce with `python corpus/sensors/fetch.py`.
Whole sweep cost $0.34.*

Three findings, two of them counterintuitive:

1. **The bigger, 3×-pricier model isn't worth it.** v4-pro's best F1 (0.811) beats
   v4-flash's best (0.809) by 0.002, for **3.1× the cost** and slower. On this task,
   `v4-flash + strict` is the pick.
2. **v4-pro hallucinates *more*, not less** — 0.21 vs 0.18. The stronger model is
   more willing to fill a not-stated field with a plausible value. Precision-only
   scoring would have hidden this and called pro the winner; the hallucination
   column is the whole reason it doesn't.
3. **Worked examples made it worse.** `strict+examples` underperforms plain
   `strict` for *both* models — the examples push over-extraction (recall up,
   precision and hallucination down). A reminder that more prompt isn't more
   accuracy, and the only way to know is to measure it.

The headline number is the hallucination rate: **~1 in 5 fields a datasheet
doesn't state gets a fabricated value**, and it goes *up* with model size. That's
the risk this harness exists to make visible.

## The corpus, and why these sensors

Eight commercial datasheets across six families — ToF, lidar, ultrasonic,
temperature, pressure, magnetometer — all **simple single-supply parts** where
every spec is a single stated number.

That choice is deliberate and it's the interesting design decision in the repo.
The first cut of this corpus used bare radar and IMU **SoCs** (a TI mmWave chip,
etc.), and they turned out to be unlabellable against a flat schema: a radar SoC
has four supply rails (no single "supply voltage"), quotes junction temperature
not ambient, and leaves range and field-of-view to the antenna you bolt on — so
half its fields have no determinate value. Chasing "what's the minimum supply
voltage of an IWR6843" through its four-rail power table is what surfaced the
rule the whole eval now rests on:

> **Null means the datasheet doesn't state it — never "the schema can't hold it"
> and never "I didn't read far enough." A value stated in a form the schema can't
> represent is a schema bug, fixed by sharpening the field, not by nulling.**

Simple sensors have one determinate answer per field, so the gold labels are
defensible and the null structure is honest. Full provenance and the labelling
conventions are in [corpus/sensors/README.md](corpus/sensors/README.md); the
conventions are also baked into the field descriptions in `schema.py`, so the
model is judged by the same rules the labels were written under.

The **worst-case-accuracy** convention does real work here: TMP117's label is
0.3 °C (the ±0.3 max over its full range), not the advertised "up to ±0.1"; and
BMP388's is 0.5 hPa absolute, not the 0.08 relative-accuracy headline — ~6× worse
than the number on the front page.

## Running it

```bash
pip install -e ".[dev]"
export DEEPSEEK_API_KEY=...          # never hardcode this; see Credentials
python corpus/sensors/fetch.py       # downloads the datasheets (needs pdftotext)

python -m datasheet_extraction evaluate --corpus corpus/sensors
python -m datasheet_extraction ablate  --corpus corpus/sensors --output results/run.json
```

`corpus/demo` (four synthetic sensors) needs no fetch and no key beyond the API
call — it's what CI and the offline tests run on.

Models are `deepseek-v4-flash` (default) and `deepseek-v4-pro`.

> **`deepseek-chat` and `deepseek-reasoner` retire on 2026-07-24.** They are
> aliases for v4-flash in non-thinking / thinking mode. This repo defaults to the
> canonical ids; the aliases still price correctly so old runs don't crash.

### Credentials

The key is read from `DEEPSEEK_API_KEY` and **has no fallback default**. A
hardcoded key in a public repo is scraped from GitHub's push firehose within
minutes, and deleting the line later doesn't remove it from git history.
`tests/test_extract.py` asserts no `sk-` string exists in `src/`.

## What DeepSeek's API forced, and how it was found out

Both came from probing the API, not assuming it behaves like OpenAI's:

**No strict schema mode.** `response_format={"type": "json_schema"}` is rejected —
*"This response_format type is unavailable now"*. So the schema travels as a
**tool's parameters** and Pydantic enforces it client-side. That's the better
channel anyway: passing the schema as the tool parameters is what carries the
field descriptions — and thus the labelling conventions — to the model. Client-side
validation is real, not a formality: `deepseek-reasoner` returns
`"field_of_view_deg": "null"` as a *quoted string*. A null-ish string is repaired
to `None` and **counted** (the `repairs` column), so the metric measures
extraction rather than JSON etiquette while the defect stays visible; a genuine
type error stays a failure.

**Forced `tool_choice` only works in non-thinking mode.** Every canonical model id
runs in thinking mode and rejects `tool_choice="required"`. `tool_choice="auto"`
is the only portable option, so a turn where the model answers in prose instead
of calling the tool is a recorded failure.

## Layout

```
src/datasheet_extraction/
  schema.py     SensorSpec — 19 fields, every one nullable; conventions in the descriptions
  prompts.py    prompt variants, as ablation arms
  extract.py    the API calls: tool-schema delivery, validation, failure capture
  compare.py    is this field right? numeric tolerance, text normalisation
  evaluate.py   outcomes -> precision / recall / F1 / hallucination rate
  cost.py       token accounting against DeepSeek's published rates
  corpus.py     loading documents + gold labels, and refusing inconsistent ones
  cli.py        extract / evaluate / ablate
corpus/demo/    four synthetic sensors + labels (offline, runs in CI)
corpus/sensors/ eight real datasheets: gold labels committed, text fetched
results/        raw ablation output
tests/          76 tests, no API key required
```

## Design notes

**Failures are data, not exceptions.** A corpus run must not lose good extractions
because one request hit a rate limit, so API errors, prose answers, malformed
JSON, and schema violations are captured per document. The evaluator scores a
failed extraction as a miss on every field — a crash is a failure, not an absence
of evidence.

**Numeric comparison is relative, not exact.** `77` and `77.0` are the same
answer; 1% tolerance absorbs rounding without absorbing real errors. Absolute
tolerance would be nonsense across fields spanning `0.01 m` to `1250 hPa`.

**Caching needs no code.** DeepSeek caches context automatically and reports the
hit/miss split per request — no `cache_control`, no minimum-prefix rules. A hit
costs ~2% of a miss, so `cost.py` bills hits and misses at their real separate
rates instead of averaging.

**Tests run without an API key.** The client is faked, so CI proves the extraction
and scoring logic without spending anything. The faked failures are the ones the
real API produces — quoted `"null"`, prose instead of a tool call — because those
were observed first.

## Known limits

- **Text in, text out.** `fetch.py` runs `pdftotext -layout`; a different
  extractor would feed the model different text, and that choice probably matters
  more than any prompt variant. Its own ablation, unbuilt.
- **Single-label ground truth**, one human verification pass. Corrections welcome —
  edit `corpus/sensors/gold.json`.
- **String matching is normalised-exact.** "I2C" vs "I2C/SPI" scores as a miss.
  Deliberate — a fuzzy comparator would inflate the numbers — but it makes
  `interface` a stricter field than it looks.
- **Prices are hardcoded** as read on 2026-07-15 and will drift.
