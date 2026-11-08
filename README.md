# Financial Statement Analysis

An interactive dashboard over SEC 10-K and 10-Q fundamentals for a watchlist
of US filers: history for one company, peer comparison, a screener, and
forensic-accounting scores.

This repository is a **published build**, not where the work happens. It is
regenerated from a private repository on each release and carries a single
commit, so there is no history to read here and pull requests have nowhere
to land.

## What the data is

Figures are extracted from XBRL filings on SEC EDGAR, which are public domain.
Every value carries its provenance — reported, derived by accounting identity,
or a trailing-twelve-month sum — and the Data Quality page breaks that down
per company. Nothing inferred is presented as reported.

Valuation metrics (P/E, EV/EBITDA, price/book and the rest) are **absent by
design**. They require share prices, and the price source used privately does
not permit redistribution. The same applies to the 1968 Altman Z-Score, whose
market-equity term depends on them; Altman's Z' and Z'' refittings use book
equity and are included.

## Running it

```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

Nothing to configure — the data ships with the repository.

## Not investment advice

A screen is a place to start looking, not a conclusion. Figures may contain
extraction errors; check anything that matters against the filing itself.
