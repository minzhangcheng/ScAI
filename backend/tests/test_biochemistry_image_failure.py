"""Biochemistry image outcomes preserve the distinction between no data and failure."""

import base64
import hashlib
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from backend.agents.biochemistry_agent import BiochemistryAgent
from backend.models.schemas import AgentType, DataFile
from backend.utils.llm_client import BadRequestError, UpstreamExhaustedError


def _agent(llm_client=None):
    agent = BiochemistryAgent.__new__(BiochemistryAgent)
    agent.agent_type = AgentType.BIOCHEMISTRY
    agent.session_id = "s"
    provider = MagicMock()
    provider.get_llm_client.return_value = llm_client
    provider.get_model.return_value = "m"
    provider.get_system_prompt.return_value = "sys"
    agent._provider = provider
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["RGBA", "LA", "P", "RGB"])
async def test_vision_receives_opaque_white_background_and_refreshes_legacy_cache(tmp_path, mode):
    image = Image.new("RGBA", (1536, 16), (0, 0, 0, 0))
    image.paste((0, 0, 0, 255), (16, 0, 32, 16))
    image.putpixel((48, 8), (0, 0, 0, 128))
    if mode == "LA":
        image = image.convert("LA")
    elif mode == "P":
        image = Image.new("P", image.size, 0)
        image.putpalette([0] * 768)
        image.paste(1, (16, 0, 32, 16))
        image.putpixel((48, 8), 2)
        image.info["transparency"] = bytes([0, 255, 128])
    elif mode == "RGB":
        image = Image.alpha_composite(
            Image.new("RGBA", image.size, "white"), image
        ).convert("RGB")
    path = tmp_path / "figure.png"
    image.save(path)
    original_bytes = path.read_bytes()
    record = DataFile(
        filename=path.name, file_path=str(path), section="biochemistry", data_type="image",
        extracted_data={
            "type": "visual_analysis_v2", "input_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "analysis": "legacy extraction", "extraction_status": "usable", "terminal": True,
        },
    )
    client = MagicMock()
    client.chat_with_vision = AsyncMock(
        return_value=SimpleNamespace(content="EXTRACTION_STATUS: usable\nfresh extraction")
    )
    agent = _agent(client)

    assert await agent._analyze_single_image(record, "review") == "EXTRACTION_STATUS: usable\nfresh extraction"
    sent = Image.open(io.BytesIO(base64.b64decode(
        client.chat_with_vision.await_args.kwargs["images"][0]
    )))
    assert sent.mode == "RGB"
    assert sent.size == image.size
    assert sent.getpixel((0, 0)) == (255, 255, 255)
    assert sent.getpixel((20, 8)) == (0, 0, 0)
    assert sent.getpixel((48, 8)) == (127, 127, 127)
    assert path.read_bytes() == original_bytes

    await agent._analyze_single_image(record, "next round")
    assert client.chat_with_vision.await_count == 1


@pytest.mark.asyncio
async def test_no_image_all_failed_and_partial_are_distinct(monkeypatch):
    agent = _agent()
    assert await agent._step2_process_images([], "instruction") is None

    monkeypatch.setattr(
        BiochemistryAgent,
        "_analyze_single_image",
        AsyncMock(side_effect=[None, "EXTRACTION_STATUS: usable\nfinding"]),
    )
    images = [
        DataFile(filename="failed.png", file_path="/nope/failed.png", section="biochemistry", data_type="image"),
        DataFile(filename="ok.png", file_path="/nope/ok.png", section="biochemistry", data_type="image"),
    ]
    agent.chat = AsyncMock(return_value=json.dumps({
        "image_findings": ["finding"],
        "observed_values": [],
        "image_mechanisms": [],
        "image_suggested_experiments": [],
    }))
    partial = await agent._step2_process_images(images, "instruction")
    assert partial["status"] == "partial"
    assert partial["failed_files"] == ["failed.png"]

    monkeypatch.setattr(BiochemistryAgent, "_analyze_single_image", AsyncMock(return_value=None))
    failed = await agent._step2_process_images(images, "instruction")
    assert failed == {"status": "failed", "failed_files": ["failed.png", "ok.png"]}


@pytest.mark.asyncio
async def test_deterministic_vision_shape_failure_is_recorded_without_hitl(tmp_path):
    from PIL import Image

    image_path = tmp_path / "biochem-shape.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    record = DataFile(
        filename=image_path.name,
        file_path=str(image_path),
        section="biochemistry",
        data_type="image",
    )
    provider_error = BadRequestError(
        message="Error code: 400 - invalid image url",
        response=MagicMock(status_code=400),
        body={"error": {
            "message": "Invalid image_url payload",
            "param": "messages[0].content[1].image_url",
            "code": 1214,
        }},
    )
    wrapped = UpstreamExhaustedError(
        "protocols exhausted", agent_name="biochemistry", model="m"
    )
    wrapped.__cause__ = provider_error
    client = MagicMock()
    client.chat_with_vision = AsyncMock(side_effect=wrapped)
    agent = _agent(client)

    assert await agent._analyze_single_image(record, "instruction") is None
    assert client.chat_with_vision.await_count == 1
    assert "protocols exhausted" in record.extracted_data["error"]
