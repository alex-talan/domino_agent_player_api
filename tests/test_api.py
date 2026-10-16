from fastapi.testclient import TestClient

from domino_player.app import create_app


class ApiProvider:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.retry_count = 0

    async def invoke_structured(self, messages, output_type):
        self.call_count += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_play_returns_exact_legal_move_and_end_has_no_body(
    settings, game_request
) -> None:
    provider = ApiProvider({"tile": 2, "position": "head"})
    with TestClient(create_app(settings, provider)) as client:
        response = client.post("/play", json=game_request)
        end_response = client.post("/end", json={"win": True})

    assert response.status_code == 200
    assert response.json() == {"tile": 2, "position": "head"}
    assert end_response.status_code == 200
    assert end_response.content == b""
    assert provider.call_count == 1


def test_invalid_input_is_400_and_never_calls_model(settings, game_request) -> None:
    provider = ApiProvider({"tile": 2, "position": "head"})
    game_request["your_tiles"] = [True]

    with TestClient(create_app(settings, provider)) as client:
        response = client.post("/play", json=game_request)

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"
    assert provider.call_count == 0


def test_end_does_not_block_later_turn_and_does_not_call_model(
    settings, game_request
) -> None:
    provider = ApiProvider({"tile": 2, "position": "head"})
    with TestClient(create_app(settings, provider)) as client:
        assert client.post("/end", json={"win": False}).status_code == 200
        assert client.post("/end", json={"win": False}).status_code == 200
        response = client.post("/play", json=game_request)

    assert response.status_code == 200
    assert provider.call_count == 1


def test_each_request_uses_its_own_authoritative_hand(settings, game_request) -> None:
    provider = ApiProvider(
        {"tile": 2, "position": "head"},
        {"tile": 11, "position": "tail"},
    )
    with TestClient(create_app(settings, provider)) as client:
        first = client.post("/play", json=game_request)
        game_request["your_tiles"] = [11]
        second = client.post("/play", json=game_request)

    assert first.json() == {"tile": 2, "position": "head"}
    assert second.json() == {"tile": 11, "position": "tail"}
    assert provider.call_count == 2


def test_history_null_for_another_player_is_accepted(settings, game_request) -> None:
    game_request["p2"] = None
    provider = ApiProvider({"tile": 2, "position": "head"})
    with TestClient(create_app(settings, provider)) as client:
        response = client.post("/play", json=game_request)

    assert response.status_code == 200
