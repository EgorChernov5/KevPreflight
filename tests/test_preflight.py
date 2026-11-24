"""Unit-тесты server preflight-контракта без загрузки весов."""
import hashlib
import queue
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from kev import checkpoint as checkpoint_module
from kev import serve
from kev.checkpoint import Checkpoint, Meta, TOKENIZER_ASSETS, write_meta
from kev.model import SERVE_MAX_STATE, SPECIAL


HUB_SHA = "a" * 40
TOKENIZER_SHA = "b" * 40


def _hub_checkpoint(tmp_path: Path, monkeypatch: object) -> Checkpoint:
    """Создаёт checkpoint, разрешённый в локальный immutable Hub snapshot."""
    snapshot = tmp_path / "models--owner--kev" / "snapshots" / HUB_SHA
    snapshot.mkdir(parents=True)
    write_meta(snapshot, Meta(base="owner/base", base_revision=TOKENIZER_SHA))
    monkeypatch.setattr(checkpoint_module, "resolve_run", lambda run: str(snapshot))
    return Checkpoint("owner/kev@main")


def _fake_server(checkpoint: Checkpoint, tokenizer: object,
                 native_window: int = 262144) -> SimpleNamespace:
    """Возвращает минимальный fake loaded Server для карточки моделей."""
    model = SimpleNamespace(
        head=SimpleNamespace(temperature=1.7), backend="torch", dtype="bfloat16",
        lm=SimpleNamespace(config=SimpleNamespace(
            max_position_embeddings=native_window)))
    return SimpleNamespace(
        checkpoint=checkpoint, tok=tokenizer, model=model, device="cuda:0",
        release_date="2026-11-24", truncate_states=False,
        prefix_cache=SimpleNamespace(
            size=4, min_tokens=0, max_tokens=65536, hits=0, misses=0,
            entries={}, oom_retries=0),
        batches=0, batched_requests=0, queue=queue.Queue())


def _card(client: TestClient) -> dict[str, object]:
    """Возвращает первую карточку ответа /v1/models."""
    return client.get("/v1/models").json()["models"][0]


def test_models_card_has_complete_preflight_and_runtime_values(
        tmp_path: Path, monkeypatch: object) -> None:
    """Карточка публикует полный контракт и runtime-поля fake Server."""
    checkpoint = _hub_checkpoint(tmp_path, monkeypatch)
    tokenizer = SimpleNamespace(
        name_or_path="owner/base", init_kwargs={"_commit_hash": TOKENIZER_SHA})
    monkeypatch.setattr(serve, "server",
                        lambda: _fake_server(checkpoint, tokenizer))

    with TestClient(serve.app) as client:
        card = _card(client)

    assert card["preflight_contract_version"] == "semantic-validation-preflight-v1"
    assert card["model_id"] == "owner/kev"
    assert card["model_revision"] == HUB_SHA
    assert card["tokenizer_revision"] == TOKENIZER_SHA
    assert card["native_window"] == 262144
    assert card["prompt_contract"] == "kev_pointer_v1_special_delimiters"
    assert card["serializer_revision"] == "kev_systemone_v1"
    assert card["special_tokens"] == SPECIAL
    assert card["prompt_reserve"] == len(SPECIAL)
    assert card["device"] == "cuda:0"
    assert card["dtype"] == "bfloat16"
    assert card["backend"] == "torch"
    assert card["backend_contract"] == "kev.http.v1.systemone"
    assert card["preflight_errors"] == []


def test_checkpoint_keeps_resolved_hub_commit(
        tmp_path: Path, monkeypatch: object) -> None:
    """Запрошенная ветка не подменяет SHA разрешённого snapshot."""
    checkpoint = _hub_checkpoint(tmp_path, monkeypatch)
    assert checkpoint.requested == "owner/kev@main"
    assert checkpoint.model_id == "owner/kev"
    assert checkpoint.model_revision == HUB_SHA


def test_tokenizer_revision_fingerprints_loaded_assets(tmp_path: Path) -> None:
    """Локальный tokenizer получает устойчивый SHA-256 своих assets."""
    assets = {
        "special_tokens_map.json": b'{"additional_special_tokens":[]}',
        "tokenizer.json": b'{"version":"1.0"}',
        "merges.txt": b"a b\nc d\n",
    }
    for name in reversed(tuple(assets)):
        (tmp_path / name).write_bytes(assets[name])
    checkpoint = Checkpoint.__new__(Checkpoint)
    checkpoint.meta = Meta(base="local")
    tokenizer = SimpleNamespace(name_or_path=str(tmp_path), init_kwargs={})

    expected = hashlib.sha256()
    for name in TOKENIZER_ASSETS:
        if name in assets:
            expected.update(f"{name}\0{len(assets[name])}\0".encode("utf-8"))
            expected.update(assets[name])
    assert checkpoint.tokenizer_revision(tokenizer) == expected.hexdigest()


def test_native_window_is_not_server_admission_limit(
        tmp_path: Path, monkeypatch: object) -> None:
    """native_window берётся из checkpoint, а max_state_tokens из сервера."""
    checkpoint = _hub_checkpoint(tmp_path, monkeypatch)
    checkpoint.meta.extra["preflight"] = {"native_window": 32768}
    tokenizer = SimpleNamespace(
        name_or_path="owner/base", init_kwargs={"_commit_hash": TOKENIZER_SHA})
    monkeypatch.setattr(
        serve, "server", lambda: _fake_server(checkpoint, tokenizer, 262144))
    with TestClient(serve.app) as client:
        card = _card(client)
    assert card["native_window"] == 32768
    assert card["max_state_tokens"] == SERVE_MAX_STATE == 65536


def test_runtime_fields_come_from_loaded_server(
        tmp_path: Path, monkeypatch: object) -> None:
    """device, dtype и backend читаются из fake loaded Server."""
    checkpoint = _hub_checkpoint(tmp_path, monkeypatch)
    tokenizer = SimpleNamespace(
        name_or_path="owner/base", init_kwargs={"_commit_hash": TOKENIZER_SHA})
    monkeypatch.setattr(serve, "server",
                        lambda: _fake_server(checkpoint, tokenizer))
    with TestClient(serve.app) as client:
        card = _card(client)
    assert (card["device"], card["dtype"], card["backend"]) == (
        "cuda:0", "bfloat16", "torch")


def test_local_checkpoint_without_provenance_returns_null_diagnostics(
        tmp_path: Path) -> None:
    """Локальный checkpoint получает fingerprint, но не придуманные tokenizer/window."""
    write_meta(tmp_path, Meta(base="local"))
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"local weights")
    checkpoint = Checkpoint(tmp_path)
    tokenizer = SimpleNamespace(name_or_path="missing", init_kwargs={})
    preflight = checkpoint.preflight(tokenizer, SimpleNamespace())
    assert str(preflight["model_id"]).startswith("checkpoint-sha256:")
    assert str(preflight["model_revision"]).startswith("sha256:")
    assert preflight["tokenizer_revision"] is None
    assert preflight["native_window"] is None
    assert len(preflight["preflight_errors"]) == 2


def test_models_preserve_legacy_card_and_aliases(
        tmp_path: Path, monkeypatch: object) -> None:
    """Прежние поля и алиасы kev-latest/jev-latest остаются без изменений."""
    checkpoint = _hub_checkpoint(tmp_path, monkeypatch)
    tokenizer = SimpleNamespace(
        name_or_path="owner/base", init_kwargs={"_commit_hash": TOKENIZER_SHA})
    monkeypatch.setattr(serve, "server",
                        lambda: _fake_server(checkpoint, tokenizer))
    with TestClient(serve.app) as client:
        cards = client.get("/v1/models").json()["models"]
    assert [card["name"] for card in cards] == ["kev-latest", "jev-latest"]
    assert {key: cards[0][key] for key in (
        "run", "base", "device", "backend", "dtype", "temperature",
        "max_state_tokens", "truncate_states")} == {
            "run": "owner/kev@main", "base": "owner/base", "device": "cuda:0",
            "backend": "torch", "dtype": "bfloat16", "temperature": 1.7,
            "max_state_tokens": 65536, "truncate_states": False}
    assert {key: value for key, value in cards[0].items() if key != "name"} == {
        key: value for key, value in cards[1].items() if key != "name"}


def test_special_tokens_have_encoding_order(
        tmp_path: Path, monkeypatch: object) -> None:
    """special_tokens всегда следуют порядку, который использует encode."""
    checkpoint = _hub_checkpoint(tmp_path, monkeypatch)
    tokenizer = SimpleNamespace(
        name_or_path="owner/base", init_kwargs={"_commit_hash": TOKENIZER_SHA})
    monkeypatch.setattr(serve, "server",
                        lambda: _fake_server(checkpoint, tokenizer))
    with TestClient(serve.app) as client:
        first = _card(client)["special_tokens"]
        second = _card(client)["special_tokens"]
    assert first == second == [
        "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>",
        "<|box_end|>", "<|fim_suffix|>"]


def test_systemone_response_is_unchanged(monkeypatch: object) -> None:
    """Preflight не меняет тело POST /v1/systemone."""
    response = {
        "model": "kev-latest",
        "answers": {"ok": {"type": "noul", "noul": 0.75}},
        "usage": {"input_tokens": 8, "output_tokens": 4},
        "latency_ms": 1.2,
    }

    class FakeServer:
        async def answer_async(self, request: object) -> dict[str, object]:
            return response

    monkeypatch.setattr(serve, "server", FakeServer)
    request = {"state": "x", "model": "kev-latest", "questions": {
        "ok": {"type": "noul", "instructions": "ok?"}}}
    with TestClient(serve.app) as client:
        result = client.post("/v1/systemone", json=request)
    assert result.status_code == 200
    assert result.json() == response
