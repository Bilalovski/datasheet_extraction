# Corpora

A corpus is a directory of `.txt` documents plus a `gold.json` mapping each
document's filename stem to its labelled fields. The evaluator scores
predictions against those labels, so the labels are the experiment — everything
else here is plumbing.

## `demo/`

Four **synthetic** datasheets, written for this repo. They are not real products
and the specifications in them are invented, which is the point: they can be
committed and redistributed, and the labels are known to be correct because the
documents were written to match them.

They exist so `pytest` runs offline and so a first `evaluate` works with nothing
but an API key. **They are not an evaluation.** Four hand-written documents
score whatever they were written to score; no conclusion about a model belongs
in a README on the strength of them.

They do cover the cases that make the metrics mean something:

| Document | What it exercises |
| --- | --- |
| `rdx-7700` | Every field stated. Min/typ/max columns, a `76 - 81 GHz` band to take the midpoint of, `±60°` half-angles to double. |
| `mmw-2440` | Unit conversion (`60 cm` into a metres field). Elevation is stated as *not specified* — filling it in is a hallucination, not a wrong answer. |
| `ls-16` | A lidar, so the radar-only fields are legitimately null. Sparse. |
| `trx-9` | A product brief with almost nothing in it. Eleven null fields: the hallucination trap. |

## Bringing your own documents

Real datasheets are copyrighted. Don't commit them — keep them out of the repo
and fetch them, the way `radar_SLAM` doesn't ship its captures:

```
corpus/real/
├── gold.json        # your labels — your work product, commit this
├── fetch.py         # downloads the PDFs from their vendor URLs
└── *.txt            # extracted text, gitignored
```

`.gitignore` already excludes `corpus/real/*.pdf` and `corpus/real/*.txt` while
keeping `gold.json`. Labels are facts about a document rather than a copy of it.

The loader takes plain text, so getting there from a PDF is your choice of
`pdftotext`, `pypdf`, or a layout-aware extractor. That choice is worth an
ablation of its own — layout mangling is likely to cost more accuracy than any
prompt variant, since a spec table flattened into prose loses the column
headers that say which number is typical and which is maximum.

## Labelling

Read the document, fill in what it states, leave the rest null. The rules the
prompt states are the rules the labels must follow, or the model is being graded
against a different spec than the one it was given:

- Fields of view are totals: `±60°` is `120`.
- A band is its midpoint: `76 - 81 GHz` is `78.5`.
- Where there are min/typ/max columns, take typical.
- Convert into the unit named in the field.
- **Null means the document doesn't state it.** Not "I couldn't find it", and
  not "it's obvious from the part number" — if it isn't in the text, it's null,
  and a model that produces it is hallucinating even when it's right.

`load_corpus` validates every label against the schema and refuses a corpus
where documents and labels don't line up, so a typo fails loudly at load rather
than quietly as a model error.
