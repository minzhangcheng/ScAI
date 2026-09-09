"""Regression tests for the direction-search defects found in the 2026-09-08 audit.

Two real runs (a local 27B model and a frontier model, same dataset) were read
end to end. Every test here fails on the tree before the accompanying fix:

- candidate labels are free text, and they were quoted verbatim as "..."[tiab]
  phrases. Inside an OR list that only wastes an arm, but tissue is ANDed into
  the direct-prior-work and prior-art queries, so a clause such as "Liver, with
  primary emphasis on hepatocytes" returned the whole query empty and the
  direction recorded zero direct evidence;
- the PI scoring prompt asked the model to echo the candidate list and the
  evidence matrix back. That is what pushed a six-candidate scoring call past
  the request deadline four times in a row, after which the round advanced with
  no direction at all and the entire literature gate was discarded;
- the scorer received evidence counts with no article identity, so it could not
  tell a real direct hit from a keyword collision;
- "known_area" was returned whenever anything at all was retrieved, and since
  triage puts most articles in the analogous bucket, that was every candidate;
- E-utilities requests were paced by one sleep placed between a single call's
  esearch and efetch. That bounds nothing when candidates are evaluated
  concurrently, and an HTTP 429 lost a whole batch silently.
"""
import asyncio
import importlib
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from backend.agents.literature_agent import LiteratureAgent
from backend.agents.literature.query_builders import normalize_search_terms
from backend.models.schemas import DirectionSearchQuery, PubMedEvidenceItem
from backend.tests.split_facade import split_source
from backend.utils.direction_payload import EVIDENCE_PROMPT_FIELDS

pubmed_module = importlib.import_module("backend.agents.literature.pubmed")

PI_SRC = split_source("backend.agents.pi_agent")


def _make_literature_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client.return_value = MagicMock()
    agent = LiteratureAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


def _evidence(pmid, title, supports, year="2024"):
    return PubMedEvidenceItem(
        pmid=pmid,
        title=title,
        year=year,
        relevance="high",
        supports=supports,
        why_it_matters="because",
    )


# --------------------------------------------------------------- term hygiene


def test_parenthesised_acronym_is_kept_and_a_qualifier_is_dropped():
    terms = normalize_search_terms(["Integrated Stress Response (ISR)"])

    assert "Integrated Stress Response" in terms
    assert "ISR" in terms
    assert not any("(" in term for term in terms)


def test_parenthesised_multiword_qualifier_leaves_only_the_head_term():
    terms = normalize_search_terms(["Hepatocyte (Stressed/Injured State)"])

    assert terms == ["Hepatocyte"]


def test_slash_separated_label_becomes_separate_searchable_terms():
    assert normalize_search_terms(["Nrf2/Antioxidant Response"]) == [
        "Nrf2",
        "Antioxidant Response",
    ]


def test_clause_too_long_to_be_an_index_term_is_dropped():
    assert normalize_search_terms(["Liver, with primary emphasis on hepatocytes"]) == [
        "Liver"
    ]
    assert normalize_search_terms(["Liver parenchymal and immune compartments"]) == []


def test_repeated_labels_collapse_to_one_term():
    terms = normalize_search_terms(
        ["Hepatocyte (Pericentral)", "Hepatocyte (Periportal)", "Hepatocyte"]
    )

    assert terms == ["Hepatocyte"]


def test_plain_terms_are_left_alone():
    assert normalize_search_terms(
        ["NRF2", "epithelial cell", "oxidative stress", "single-cell RNA-seq"]
    ) == ["NRF2", "epithelial cell", "oxidative stress", "single-cell RNA-seq"]


def test_candidate_terms_normalise_tissue_to_a_searchable_phrase():
    terms = _make_literature_agent()._candidate_terms(
        {
            "key_genes": ["Mt1"],
            "key_pathways": ["Integrated Stress Response (ISR)"],
            "key_cell_types": ["Hepatocyte (Pericentral)", "Hepatocyte (Periportal)"],
            "organism": "Mus musculus",
            "tissue": "Liver, with primary emphasis on hepatocytes",
        }
    )

    assert terms["tissue"] == ["Liver"]
    assert terms["organism"] == ["Mus musculus"]
    assert terms["cells"] == ["Hepatocyte"]
    assert "ISR" in terms["pathways"]


def test_unindexable_tissue_is_omitted_instead_of_emptying_the_query():
    agent = _make_literature_agent()
    terms = agent._candidate_terms(
        {
            "key_genes": ["Slc31a1", "Atp7b"],
            "organism": "Mus musculus",
            "tissue": "Liver parenchymal and immune compartments",
        }
    )

    query = agent._build_direct_prior_work(terms, deep=False)

    assert terms["tissue"] == [""]
    assert query
    assert "parenchymal" not in query
    assert "Slc31a1" in query
    assert "Mus musculus" in query


# ------------------------------------------------------- scoring prompt budget


def _scoring_prompt_block() -> str:
    return PI_SRC.split("def score_directions_with_evidence", 1)[1].split(
        "async def", 1
    )[0]


def test_scoring_prompt_does_not_ask_the_model_to_echo_its_own_input():
    block = _scoring_prompt_block()

    assert "echo the raw candidates" not in block
    assert "echo the evidence matrix" not in block


def test_scoring_prompt_states_that_carried_fields_must_not_be_repeated():
    block = _scoring_prompt_block()

    assert "Return ONLY the fields shown above" in block
    assert "merged onto your entry automatically" in block


def test_scoring_output_schema_keeps_the_scored_fields_and_drops_the_narrative_ones():
    template = _scoring_prompt_block().split('"research_direction_candidates"', 1)[
        1
    ].split('"direction_dependency_graph"', 1)[0]

    for field in ("direction_id", "overall_priority", "evidence_ref", "functional_impact"):
        assert field in template, f"scored field missing from schema: {field}"
    for field in ("core_observation", "main_hypothesis", "alternative_hypotheses",
                  "supporting_data"):
        assert f'"{field}"' not in template, f"schema still asks the model to restate {field}"


def test_integration_scoring_prompt_asks_only_for_the_new_fields():
    block = PI_SRC.split("def score_integration_directions_with_evidence", 1)[1].split(
        "async def", 1
    )[0]

    assert "keep every field it already has" not in block
    assert "ONLY the fields below" in block


# ---------------------------------------------------------- evidence identity


def test_key_evidence_names_the_articles_behind_the_counts():
    agent = _make_literature_agent()

    rendered = agent._build_key_evidence(
        [
            _evidence("111", "Copper deficiency blocks Nrf2 in mouse liver", "direct"),
            _evidence("222", "Hepatic Ctr1 deletion", "direct"),
            _evidence("333", "Third", "direct"),
            _evidence("444", "Fourth", "direct"),
        ],
        [_evidence("555", "Resistance to copper toxicity", "contradictory")],
    )

    assert "PMID 111" in rendered
    assert "Copper deficiency blocks Nrf2 in mouse liver" in rendered
    assert "(+1 more)" in rendered
    assert "contradictory: PMID 555" in rendered


def test_key_evidence_is_empty_when_there_is_none():
    assert _make_literature_agent()._build_key_evidence([], []) == ""


def test_scoring_prompt_receives_key_evidence():
    # The field has to be in the prompt projection, and ahead of the prose
    # fields, or the per-record budget drops it before the scorer sees it.
    assert "key_evidence" in EVIDENCE_PROMPT_FIELDS
    assert EVIDENCE_PROMPT_FIELDS.index("key_evidence") < EVIDENCE_PROMPT_FIELDS.index(
        "summary"
    )


def test_scoring_rules_tell_the_pi_to_read_the_named_articles():
    assert "key_evidence names the articles" in _scoring_prompt_block()


# -------------------------------------------------------------- novelty status


def test_related_work_alone_is_not_reported_as_a_known_area():
    status = _make_literature_agent()._resolve_novelty_status(
        novelty="medium",
        timed_out=False,
        confidence="medium",
        direct_count=0,
        analogous_count=44,
        neg_conf="low",
    )

    assert status == "adjacent_work_only"


def test_direct_prior_work_is_still_a_known_area():
    status = _make_literature_agent()._resolve_novelty_status(
        novelty="low",
        timed_out=False,
        confidence="high",
        direct_count=2,
        analogous_count=10,
        neg_conf="low",
    )

    assert status == "known_area"


def test_a_timeout_still_outranks_the_bucket_counts():
    status = _make_literature_agent()._resolve_novelty_status(
        novelty="medium",
        timed_out=True,
        confidence="medium",
        direct_count=0,
        analogous_count=5,
        neg_conf="low",
    )

    assert status == "unknown_due_to_timeout"


def test_scoring_prompt_explains_the_adjacent_work_verdict():
    assert "novelty_status=adjacent_work_only" in _scoring_prompt_block()


# ------------------------------------------------------------ E-utilities rate


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self.request = httpx.Request(
            "GET", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        )

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=self.request, response=self  # type: ignore[arg-type]
            )


def _install_fake_transport(monkeypatch, script):
    calls = []

    class _FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, params=None):
            calls.append({"url": url, "params": dict(params or {}), "at": time.monotonic()})
            if script:
                return script.pop(0)
            return _FakeResponse(payload={"esearchresult": {"idlist": []}})

    monkeypatch.setattr(
        pubmed_module,
        "httpx",
        SimpleNamespace(AsyncClient=_FakeClient, HTTPStatusError=httpx.HTTPStatusError),
    )
    return calls


def test_shipped_rate_matches_the_ncbi_ceiling_without_a_key():
    # An API key is optional and only lifts the ceiling; email/tool identify the
    # caller and grant nothing. The default must be the no-key limit.
    assert pubmed_module.NCBI_RATE_LIMIT_PER_SECOND == 3.0
    assert pubmed_module.NCBI_RATE_LIMIT_PER_SECOND_WITH_KEY == 10.0


@pytest.mark.asyncio
async def test_requests_identify_the_caller_and_send_no_api_key_by_default(monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    calls = _install_fake_transport(
        monkeypatch,
        [
            _FakeResponse(payload={"esearchresult": {"idlist": ["1"]}}),
            _FakeResponse(text="<PubmedArticleSet></PubmedArticleSet>"),
        ],
    )
    agent = _make_literature_agent()
    agent.pubmed_email = "someone@example.org"

    await agent._pubmed_search("copper deficiency", max_results=5)

    assert len(calls) == 2
    for call in calls:
        assert call["params"]["tool"] == "ScAI"
        assert call["params"]["email"] == "someone@example.org"
        assert "api_key" not in call["params"]


@pytest.mark.asyncio
async def test_an_api_key_in_the_environment_is_sent_and_raises_the_rate(monkeypatch):
    monkeypatch.setenv("NCBI_API_KEY", "abc123")
    calls = _install_fake_transport(
        monkeypatch, [_FakeResponse(payload={"esearchresult": {"idlist": []}})]
    )
    agent = _make_literature_agent()

    await agent._pubmed_search("copper deficiency")

    assert calls[0]["params"]["api_key"] == "abc123"
    assert agent._entrez_api_key() == "abc123"


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_request_gate(monkeypatch):
    # Candidates are evaluated concurrently. A per-call sleep leaves the request
    # rate proportional to the number of tasks, which is how the gate collected
    # 429s that surfaced only as a missing evidence axis.
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.setattr(pubmed_module, "NCBI_RATE_LIMIT_PER_SECOND", 10.0)
    calls = _install_fake_transport(monkeypatch, [])
    agent = _make_literature_agent()

    await asyncio.gather(*(agent._pubmed_search(f"q{i}") for i in range(4)))

    assert len(calls) == 4
    stamps = sorted(call["at"] for call in calls)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert min(gaps) >= 0.08, f"requests were not paced: {gaps}"


@pytest.mark.asyncio
async def test_a_rate_limit_rejection_is_retried_instead_of_losing_the_batch(monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.setattr(pubmed_module, "NCBI_RATE_LIMIT_BACKOFF_SECONDS", 0.01)
    calls = _install_fake_transport(
        monkeypatch,
        [
            _FakeResponse(status_code=429),
            _FakeResponse(payload={"esearchresult": {"idlist": ["7"]}}),
            _FakeResponse(text="<PubmedArticleSet></PubmedArticleSet>"),
        ],
    )
    agent = _make_literature_agent()

    articles = await agent._pubmed_search("copper deficiency")

    assert len(calls) == 3, "the rejected request was not retried"
    assert articles == []


@pytest.mark.asyncio
async def test_a_persistent_rate_limit_raises_rather_than_reporting_no_results(monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.setattr(pubmed_module, "NCBI_RATE_LIMIT_BACKOFF_SECONDS", 0.01)
    _install_fake_transport(monkeypatch, [_FakeResponse(status_code=429)] * 8)
    agent = _make_literature_agent()

    with pytest.raises(httpx.HTTPStatusError):
        await agent._pubmed_search("copper deficiency")


# --------------------------------------------- triage cost and evidence identity
#
# Measured on two 0.55.9 runs of the direction gate: a fifth of every article
# sent to the triage model had already been triaged for the same direction under
# a different query category, and _dedup_evidence keeps only the first verdict
# per PMID, so the repeat was paid for and thrown away. Separately, key_evidence
# rendered only the direct and contradictory buckets, which left it empty for
# exactly the adjacent_work_only directions -- the ones carrying 57 to 77
# analogous articles and no direct hit.


def _ev(pmid, supports, title="T"):
    return PubMedEvidenceItem(
        pmid=pmid, title=title, year="2024", relevance="high",
        supports=supports, why_it_matters="because",
    )


@pytest.mark.asyncio
async def test_an_article_is_not_sent_to_triage_twice_for_one_direction(monkeypatch):
    agent = _make_literature_agent()
    triaged_batches = []

    async def fake_search(query, max_results=20):
        return [{"pmid": "111", "title": "a", "abstract": ""},
                {"pmid": "222", "title": "b", "abstract": ""}]

    async def fake_triage(articles, candidate, query):
        triaged_batches.append([a["pmid"] for a in articles])
        return [], {"off_topic": False, "contradictory_ambiguous": False,
                    "species_tissue_qualification": "", "summary": ""}

    agent._pubmed_search = fake_search
    agent._triage_articles_with_llm = fake_triage

    seen = set()
    q1 = DirectionSearchQuery(category="direct_prior_work", query="q1")
    q2 = DirectionSearchQuery(category="pathway_focused", query="q2")
    for q in (q1, q2):
        await agent._execute_query_batch(q, {}, [], 30.0, triaged_pmids=seen)

    assert triaged_batches[0] == ["111", "222"]
    assert len(triaged_batches) == 1, "the second category re-sent articles already triaged"
    assert seen == {"111", "222"}


@pytest.mark.asyncio
async def test_only_the_repeated_articles_are_withheld(monkeypatch):
    agent = _make_literature_agent()
    batches = []
    pages = iter([
        [{"pmid": "111", "title": "a", "abstract": ""}],
        [{"pmid": "111", "title": "a", "abstract": ""},
         {"pmid": "333", "title": "c", "abstract": ""}],
    ])

    async def fake_search(query, max_results=20):
        return next(pages)

    async def fake_triage(articles, candidate, query):
        batches.append([a["pmid"] for a in articles])
        return [], {}

    agent._pubmed_search = fake_search
    agent._triage_articles_with_llm = fake_triage

    seen = set()
    for cat in ("direct_prior_work", "pathway_focused"):
        await agent._execute_query_batch(
            DirectionSearchQuery(category=cat, query="q"), {}, [], 30.0,
            triaged_pmids=seen,
        )

    assert batches == [["111"], ["333"]]


@pytest.mark.asyncio
async def test_deduplication_is_off_when_no_set_is_supplied():
    # Other callers and older tests must keep their behaviour exactly.
    agent = _make_literature_agent()
    batches = []

    async def fake_search(query, max_results=20):
        return [{"pmid": "111", "title": "a", "abstract": ""}]

    async def fake_triage(articles, candidate, query):
        batches.append([a["pmid"] for a in articles])
        return [], {}

    agent._pubmed_search = fake_search
    agent._triage_articles_with_llm = fake_triage

    for cat in ("direct_prior_work", "pathway_focused"):
        await agent._execute_query_batch(
            DirectionSearchQuery(category=cat, query="q"), {}, [], 30.0
        )

    assert batches == [["111"], ["111"]]


def test_key_evidence_names_adjacent_articles_when_there_is_no_direct_hit():
    agent = _make_literature_agent()

    rendered = agent._build_key_evidence(
        [], [], [_ev("111", "analogous", "Interferon signalling in mouse liver")]
    )

    assert "adjacent (not direct)" in rendered
    assert "PMID 111" in rendered


def test_adjacent_articles_are_not_added_beside_direct_evidence():
    agent = _make_literature_agent()

    rendered = agent._build_key_evidence(
        [_ev("111", "direct")], [], [_ev("222", "analogous")]
    )

    assert "PMID 111" in rendered
    assert "PMID 222" not in rendered


def test_key_evidence_is_still_empty_when_nothing_was_retrieved():
    assert _make_literature_agent()._build_key_evidence([], [], []) == ""


# ------------------------------------------- challenge anchoring and study context
#
# Measured over the 8 real direction candidates from two runs, against live
# PubMed. The unanchored challenge query returned 159 articles of which 11 were
# in the study's organism and tissue; anchored it returns 17, all 17 of them in
# organism and tissue. Six of sixteen challenge queries now return nothing at
# all, which is the honest outcome: before, they returned twenty off-topic
# articles that triaged to no contradiction.
#
# Separately, adding the study condition as another mandatory AND block was
# measured and REJECTED: `(genes) AND organism AND tissue AND (condition)`
# returned zero articles for all eight candidates. The condition belongs in the
# triage decision, where it costs no extra search.


def _terms(agent, **over):
    candidate = {
        "key_genes": ["Mt1", "Mt2"],
        "key_pathways": ["oxidative stress"],
        "organism": "Mus musculus",
        "tissue": "Liver",
    }
    candidate.update(over)
    return agent._candidate_terms(candidate)


def test_the_challenge_query_is_anchored_to_organism_and_tissue():
    agent = _make_literature_agent()

    q = agent._build_active_challenge_query(_terms(agent))

    assert '"Mus musculus"[all]' in q
    assert '"Liver"[tiab]' in q
    assert "contradict*[tiab]" in q


def test_the_contradictory_expectedness_query_is_anchored_too():
    agent = _make_literature_agent()

    q = agent._build_contradictory_expectedness(_terms(agent))

    assert '"Mus musculus"[all]' in q
    assert '"Liver"[tiab]' in q


def test_the_anchor_is_omitted_when_the_candidate_names_no_organism_or_tissue():
    agent = _make_literature_agent()

    q = agent._build_active_challenge_query(_terms(agent, organism="", tissue=""))

    assert q
    assert "[all]" not in q
    assert "Mt1" in q


@pytest.mark.asyncio
async def test_the_triage_prompt_states_the_study_under_investigation():
    agent = _make_literature_agent()
    agent._direction_study_context = "Hepatocyte-specific ZnT1/CTR1 double knockout"
    captured = {}

    async def fake_chat(prompt, temperature=0.1):
        captured["p"] = prompt
        return '{"items": []}', False

    agent._chat_with_fallback = fake_chat

    await agent._triage_articles_with_llm(
        [{"pmid": "1", "title": "T", "abstract": "A"}],
        {"direction": "d", "key_genes": ["Mt1"], "organism": "Mus musculus", "tissue": "Liver"},
        DirectionSearchQuery(category="direct_prior_work", query="q"),
    )

    assert "Study under investigation: Hepatocyte-specific ZnT1/CTR1 double knockout" in captured["p"]


@pytest.mark.asyncio
async def test_the_triage_prompt_says_so_when_no_study_context_is_available():
    agent = _make_literature_agent()
    captured = {}

    async def fake_chat(prompt, temperature=0.1):
        captured["p"] = prompt
        return '{"items": []}', False

    agent._chat_with_fallback = fake_chat

    await agent._triage_articles_with_llm(
        [{"pmid": "1", "title": "T", "abstract": "A"}], {"direction": "d"},
        DirectionSearchQuery(category="direct_prior_work", query="q"),
    )

    assert "Study under investigation: not supplied" in captured["p"]


@pytest.mark.asyncio
async def test_the_same_gene_under_a_different_perturbation_is_not_direct():
    # A melatonin-receptor MT2 paper came back as direct prior art for a
    # metallothionein direction, because the bucket definition asked for the
    # experimental context and the prompt never supplied it.
    agent = _make_literature_agent()
    captured = {}

    async def fake_chat(prompt, temperature=0.1):
        captured["p"] = prompt
        return '{"items": []}', False

    agent._chat_with_fallback = fake_chat

    await agent._triage_articles_with_llm(
        [{"pmid": "1", "title": "T", "abstract": "A"}], {"direction": "d"},
        DirectionSearchQuery(category="direct_prior_work", query="q"),
    )

    assert "is NOT direct" in captured["p"]


@pytest.mark.asyncio
async def test_the_study_context_comes_from_the_objective_not_the_candidate_field():
    # One observed run filled experimental_context with
    # "2 group(s): Con (3 sample(s)), DKO (3 sample(s))" for every candidate,
    # which names no perturbation at all.
    agent = _make_literature_agent()
    state = SimpleNamespace(
        initial_instruction="Hepatocyte-specific ZnT1/CTR1 double knockout mice",
        literature_reports=[],
    )

    await agent.evaluate_direction_novelty_candidates([], state=state)

    assert agent._direction_study_context == "Hepatocyte-specific ZnT1/CTR1 double knockout mice"
