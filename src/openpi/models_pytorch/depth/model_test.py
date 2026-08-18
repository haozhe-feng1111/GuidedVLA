import types

import torch

from openpi.models_pytorch.depth import model as depth_model


class _FakeDINOv2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(model_type="dinov2", hidden_size=768)
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, *, pixel_values, output_hidden_states, return_dict):
        assert pixel_values.shape == (1, 3, 224, 224)
        assert output_hidden_states is True
        assert return_dict is True
        hidden_states = tuple(torch.randn(1, 257, 768) for _ in range(13))
        return types.SimpleNamespace(hidden_states=hidden_states)


def test_dinov2_base_feature_contract(monkeypatch):
    fake_model = _FakeDINOv2()

    def load_model(path, *, local_files_only, trust_remote_code):
        assert path == "/models/dinov2-base"
        assert local_files_only is True
        assert trust_remote_code is False
        return fake_model

    monkeypatch.setattr(depth_model.AutoModel, "from_pretrained", load_model)
    monkeypatch.setattr(torch, "compile", lambda module, **_: module)

    encoder = depth_model.DepthEncoder(
        "/models/dinov2-base",
        depth_encoder_type="dinov2_base",
    )
    outputs = encoder(torch.zeros(1, 3, 224, 224))

    assert len(outputs) == 4
    assert all(output.shape == (1, 16, 1024) for output in outputs)
    assert all(not parameter.requires_grad for parameter in fake_model.parameters())
