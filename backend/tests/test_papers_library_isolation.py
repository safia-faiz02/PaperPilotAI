# Tests for the shared-paper-pool / per-user-library design (papers.py):
# the same arXiv paper found by two users should be stored once, but each
# user should only ever see what's in THEIR OWN library. These mock out
# arXiv (search_arxiv) and embedding (embed_paper) since neither should
# require real network calls or a live Qdrant instance to test ownership
# logic.

from unittest.mock import AsyncMock, patch

from tests._helpers import auth_headers_for

FAKE_PAPER = {
    "external_id": "1234.5678",
    "source": "arxiv",
    "title": "Fake Paper One",
    "authors": ["A. Author"],
    "abstract": "An abstract about testing.",
    "year": 2024,
    "venue": "arXiv",
    "citation_count": None,
}


def _search(client, headers, query="testing"):
    return client.post("/papers/search", json={"query": query, "max_results": 5}, headers=headers)


@patch("app.api.routes.papers.search_arxiv", new_callable=AsyncMock)
def test_search_saves_paper_and_adds_it_to_the_searcher_s_library(mock_search_arxiv, client):
    mock_search_arxiv.return_value = [FAKE_PAPER]
    headers = auth_headers_for(client, "usera@example.com")

    response = _search(client, headers)
    assert response.status_code == 200
    assert response.json()["new_papers"] == 1

    library = client.get("/papers/", headers=headers).json()
    assert len(library) == 1
    assert library[0]["title"] == "Fake Paper One"
    assert library[0]["is_embedded"] is False


@patch("app.api.routes.papers.search_arxiv", new_callable=AsyncMock)
def test_second_user_does_not_see_first_user_s_library(mock_search_arxiv, client):
    mock_search_arxiv.return_value = [FAKE_PAPER]
    headers_a = auth_headers_for(client, "usera@example.com")
    headers_b = auth_headers_for(client, "userb@example.com")

    _search(client, headers_a)

    assert client.get("/papers/", headers=headers_b).json() == []


@patch("app.api.routes.papers.search_arxiv", new_callable=AsyncMock)
def test_same_paper_deduplicated_across_users_but_libraries_stay_separate(mock_search_arxiv, client):
    mock_search_arxiv.return_value = [FAKE_PAPER]
    headers_a = auth_headers_for(client, "usera@example.com")
    headers_b = auth_headers_for(client, "userb@example.com")

    response_a = _search(client, headers_a)
    response_b = _search(client, headers_b)

    # Same underlying paper row reused for the second user...
    assert response_a.json()["new_papers"] == 1
    assert response_b.json()["new_papers"] == 0
    assert response_b.json()["already_existed"] == 1

    # ...but it shows up in both users' own libraries independently.
    library_a = client.get("/papers/", headers=headers_a).json()
    library_b = client.get("/papers/", headers=headers_b).json()
    assert len(library_a) == 1
    assert len(library_b) == 1
    assert library_a[0]["id"] == library_b[0]["id"]


@patch("app.api.routes.papers.search_arxiv", new_callable=AsyncMock)
def test_get_paper_404s_for_a_paper_not_in_your_library(mock_search_arxiv, client):
    mock_search_arxiv.return_value = [FAKE_PAPER]
    headers_a = auth_headers_for(client, "usera@example.com")
    headers_b = auth_headers_for(client, "userb@example.com")

    paper_id = _search(client, headers_a).json()["papers"][0]["id"]

    assert client.get(f"/papers/{paper_id}", headers=headers_b).status_code == 404
    assert client.get(f"/papers/{paper_id}", headers=headers_a).status_code == 200


@patch("app.api.routes.papers.embed_paper")
@patch("app.api.routes.papers.search_arxiv", new_callable=AsyncMock)
def test_embed_persists_is_embedded_flag(mock_search_arxiv, mock_embed_paper, client):
    mock_search_arxiv.return_value = [FAKE_PAPER]
    mock_embed_paper.return_value = True
    headers = auth_headers_for(client, "usera@example.com")

    paper_id = _search(client, headers).json()["papers"][0]["id"]
    embed_response = client.post(f"/papers/{paper_id}/embed", headers=headers)
    assert embed_response.status_code == 200

    paper = client.get(f"/papers/{paper_id}", headers=headers).json()
    assert paper["is_embedded"] is True


@patch("app.api.routes.papers.embed_paper")
@patch("app.api.routes.papers.search_arxiv", new_callable=AsyncMock)
def test_embed_all_only_touches_the_current_user_s_library(mock_search_arxiv, mock_embed_paper, client):
    mock_embed_paper.return_value = True
    headers_a = auth_headers_for(client, "usera@example.com")
    headers_b = auth_headers_for(client, "userb@example.com")

    mock_search_arxiv.return_value = [FAKE_PAPER]
    _search(client, headers_a)

    other_paper = {**FAKE_PAPER, "external_id": "9999.0001", "title": "Fake Paper Two"}
    mock_search_arxiv.return_value = [other_paper]
    _search(client, headers_b, query="other topic")

    result_b = client.post("/papers/embed-all", headers=headers_b).json()
    assert result_b["embedded_count"] == 1

    # User A's paper is untouched by User B's embed-all.
    library_a = client.get("/papers/", headers=headers_a).json()
    assert library_a[0]["is_embedded"] is False


@patch("app.api.routes.papers.search_arxiv", new_callable=AsyncMock)
def test_remove_from_library_deletes_only_the_current_user_s_entry(mock_search_arxiv, client):
    mock_search_arxiv.return_value = [FAKE_PAPER]
    headers_a = auth_headers_for(client, "usera@example.com")
    headers_b = auth_headers_for(client, "userb@example.com")

    paper_id = _search(client, headers_a).json()["papers"][0]["id"]
    _search(client, headers_b)  # User B also has it in their library

    remove_response = client.delete(f"/papers/{paper_id}", headers=headers_a)
    assert remove_response.status_code == 204

    assert client.get(f"/papers/{paper_id}", headers=headers_a).status_code == 404
    # User B's library entry survives — removing is per-user, not global.
    assert client.get(f"/papers/{paper_id}", headers=headers_b).status_code == 200


def test_remove_from_library_404s_for_a_paper_not_in_your_library(client):
    headers = auth_headers_for(client, "usera@example.com")
    response = client.delete("/papers/999", headers=headers)
    assert response.status_code == 404
