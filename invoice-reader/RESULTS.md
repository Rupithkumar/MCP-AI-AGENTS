# Invoice reader: results

150 synthetic invoices (clean, mild and harsh scans). Real invoices will score lower: see Limitations.

## Headline

| System | Docs fully correct | Flagged for review | Silent errors | Accuracy of accepted docs | Errors caught | LLM calls |
|---|---|---|---|---|---|---|
| rules | 79.3% | 18.0% | 2.7% | 96.7% | 87% | 0 |
| rules_repair | 80.7% | 16.7% | 2.7% | 96.8% | 86% | 0 |

*Silent error* = a wrong document the self-check did not flag, so nobody reviews it. That is the number to minimise.
*Flagged* documents go to a human review queue.

## Accuracy per field

| System | vendor | invoice_number | date | currency | subtotal | tax | total |
|---|---|---|---|---|---|---|---|
| rules | 99% | 99% | 100% | 85% | 93% | 97% | 98% |
| rules_repair | 99% | 99% | 100% | 85% | 96% | 98% | 97% |

## Accuracy by scan quality (docs fully correct)

| System | clean | mild | harsh |
|---|---|---|---|
| rules | 80% | 86% | 72% |
| rules_repair | 82% | 88% | 72% |

## Where the errors are

- **rules**: currency (22), subtotal (11), tax (4), total (3), invoice_number (2), vendor (2)
- **rules_repair**: currency (22), subtotal (6), total (4), tax (3), invoice_number (2), vendor (2)

## Limitations

- Synthetic invoices with 3 layouts. Real invoices have far more variety, stamps, tables and handwriting.
- Text-only pipeline (OCR then extraction). A vision-language model reading the image directly would avoid OCR mistakes such as the rupee sign.
- Rules were developed while looking at one generated set; the numbers above are from a separate set with a different seed.
- The arithmetic check can only catch errors that break the totals. A garbled vendor name or invoice number passes it.
