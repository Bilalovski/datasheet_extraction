# Real datasheet corpus (8 sensors)

Eight commercial sensor datasheets with hand-verified gold labels. This is the
corpus the headline numbers in the top-level README are measured on.

## What's here, and what isn't

`gold.json` (the labels) is committed. The datasheet **text and PDFs are not** —
they're copyrighted, so redistributing them in a public repo isn't ok. Instead:

```bash
python corpus/sensors/fetch.py     # downloads the 8 PDFs, extracts text, deletes the PDFs
```

`fetch.py` needs `pdftotext` (poppler-utils). It writes `<doc_id>.txt` next to
`gold.json`, which is what the loader reads. Both are gitignored.

## The sensors

Deliberately *simple* parts — single supply, single-number specs — spanning six
families so the schema's cross-family null structure gets exercised:

| doc_id | part | family |
| --- | --- | --- |
| `vl53l1x` | ST VL53L1X | ToF |
| `vl53l0x` | ST VL53L0X | ToF |
| `lidarlite-v3` | Garmin LIDAR-Lite v3 | lidar |
| `hc-sr04` | HC-SR04 | ultrasonic |
| `tmp117` | TI TMP117 | temperature |
| `bmp388` | Bosch BMP388 | pressure |
| `bmm150` | Bosch BMM150 | magnetometer |
| `bme680` | Bosch BME680 | environmental |

Bare radar/IMU SoCs were deliberately excluded: their multi-rail supplies,
junction-not-ambient temperatures, and application-dependent ranges have no
single determinate value, so they can't be labelled cleanly against a flat
schema. See the top-level README's design notes.

## Labelling conventions

These decide what the "right" answer is, so the model is told the same rules (the
field descriptions in `schema.py` carry them verbatim):

- **Values come from the parametric tables** (Recommended Operating Conditions,
  Electrical Characteristics), not the marketing summary on page 1.
- **Accuracy is worst-case** over the full range — the `±X max` figure, not the
  best-case headline. (TMP117 → 0.3 °C, not the advertised 0.1.)
- **Pressure accuracy is absolute**, not the smaller relative-accuracy headline.
  (BMP388 → 0.5 hPa absolute, not 0.08 relative.)
- **Supply is the single main rail you provide** (the VDD you plug in); a single
  stated nominal fills both min and max.
- **Field of view is one symmetric-cone angle** — no azimuth/elevation split.
- **Magnetic range is the max full-scale across axes.** (BMM150 → 2500 µT, the
  z-axis; x/y is 1300.)
- **Null means the datasheet doesn't state it** — never "the schema can't hold
  it" and never "I didn't read far enough." A value stated in a form the schema
  can't represent is a schema problem, fixed by sharpening the field, not by
  nulling.

## Verifying / correcting a label

Open the datasheet (run `fetch.py`, or grab the PDF from the URL in that script)
and check the value against the cited spec table. The labels were drafted with
per-field provenance and one human verification pass; corrections are welcome —
edit `gold.json` and the numbers move with it.
