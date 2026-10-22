"""Clause extraction and source windows of the citation check (reticle.agent.consult); no model calls."""
from reticle.agent.consult import claim_clauses, source_window


def test_each_marker_is_judged_on_its_own_clause():
    part = 'SLN biopsy is the standard initial axillary staging procedure [E4.1] and causes less morbidity than ALND [E4.4].'
    assert claim_clauses(part) == [('SLN biopsy is the standard initial axillary staging procedure', ['E4.1']),
                                   ('and causes less morbidity than ALND', ['E4.4'])]


def test_adjacent_markers_share_a_clause_and_semicolons_start_a_new_one():
    part = 'Completion ALND may be omitted in selected patients; axillary radiotherapy is an accepted alternative [E9.1][E9.2].'
    assert claim_clauses(part) == [('axillary radiotherapy is an accepted alternative', ['E9.1', 'E9.2'])]


def test_long_sources_are_shown_where_they_match_the_claim():
    table = 'pN0 No regional lymph node metastasis identified. ' * 30 + 'pN1mi Micrometastases (~200 cells, >0.2 mm, but ≤2.0 mm). ' + 'pN2 four to nine nodes. ' * 30
    w = source_window('pN1mi: micrometastasis >0.2 mm but ≤2.0 mm', table)
    assert 'pN1mi Micrometastases' in w and len(w) <= 702 and w.startswith('…')
    assert source_window('short claim', 'A short source sentence.') == 'A short source sentence.'
