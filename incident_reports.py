# ---------------------------------------------------------------------------
# INCIDENT REPORTS — document corpus / knowledge base
#
# These are the records the RAG system is grounded in. In mini_rag.py the
# same strings are defined inline; this file exists so other scripts can
# import them by name rather than duplicating the raw text.
#
# In a real system this data would live in a database or log store and be
# loaded at ingestion time. Keeping it here as named variables makes the
# content easy to read, diff, and expand without touching pipeline logic.
# ---------------------------------------------------------------------------

report_1 = "Pump 7 experienced failure during a freeze event on 2026-02-10. Temperature dropped below 28°F. Ice formation blocked intake valve. Manual thaw required."

report_2 = "Pump 3 failed in summer due to overheating. Cooling fan malfunctioned. Internal temperature exceeded threshold."

report_3 = "Pump 7 maintenance log. Winter inspection completed. Anti-freeze system tested and functional."
