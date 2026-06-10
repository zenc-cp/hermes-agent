"""Tests for zenbrain#64 — azure-foundry must honor ``auth_mode: entra_id``.

When a fallback chain entry uses::

    fallback_model:
      model: gpt-4o
      provider: azure-foundry
      auth_mode: entra_id
      base_url: https://<foundry>.cognitiveservices.azure.com

hermes must:

1. Read ``auth_mode`` from the per-fallback config (not just the top-level
   ``model:`` block) and honor ``entra_id``.
2. When ``base_url`` is a bare ``cognitiveservices.azure.com`` host,
   default the Entra token scope to
   ``https://cognitiveservices.azure.com/.default`` and construct the
   Azure-OpenAI deployment path
   ``{base_url}/openai/deployments/{model}/chat/completions?api-version=<v>``
   (api-version configurable, default ``2024-10-21``).
3. Accept an optional ``client_id`` field to pin a user-assigned
   managed identity via
   ``DefaultAzureCredential(managed_identity_client_id=...)``.
4. Cache the credential per ``EntraIdentityConfig`` so a second call
   within the token's lifetime does not re-instantiate it.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures — stub azure.identity so CI stays hermetic.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_credential_cache():
    from agent.azure_identity_adapter import reset_credential_cache
    reset_credential_cache()
    yield
    reset_credential_cache()


@pytest.fixture
def fake_azure_identity(monkeypatch):
    """Records credential kwargs + tracks get_token calls."""
    from agent import azure_identity_adapter as _adapter

    records = {"credential_kwargs": [], "get_token_scopes": [], "scope": None}

    class _FakeCredential:
        def __init__(self, **kwargs):
            records["credential_kwargs"].append(kwargs)

        def get_token(self, scope):
            records["get_token_scopes"].append(scope)
            return SimpleNamespace(token="fake-jwt", expires_on=9999999999)

    def _bearer_provider(credential, scope):
        records["scope"] = scope

        def _mint():
            return credential.get_token(scope).token

        return _mint

    fake_module = SimpleNamespace(
        DefaultAzureCredential=_FakeCredential,
        get_bearer_token_provider=_bearer_provider,
    )
    monkeypatch.setattr(_adapter, "_require_azure_identity", lambda: fake_module)
    monkeypatch.setitem(sys.modules, "azure.identity", fake_module)
    return records


# ---------------------------------------------------------------------------
# Cycle 1 — ``client_id`` config field for user-assigned MI
# ---------------------------------------------------------------------------


class TestEntraIdentityConfigClientId:
    def test_client_id_field_defaults_to_empty(self):
        from agent.azure_identity_adapter import EntraIdentityConfig

        cfg = EntraIdentityConfig()
        assert cfg.client_id == ""

    def test_client_id_passed_to_default_azure_credential(self, fake_azure_identity):
        """User-assigned managed identity is pinned via the
        ``managed_identity_client_id`` kwarg on DefaultAzureCredential
        (Microsoft-documented contract for IMDS user-assigned MI)."""
        from agent.azure_identity_adapter import (
            EntraIdentityConfig,
            build_credential,
        )

        cfg = EntraIdentityConfig(client_id="11111111-2222-3333-4444-555555555555")
        build_credential(cfg)

        kwargs_list = fake_azure_identity["credential_kwargs"]
        assert kwargs_list, "DefaultAzureCredential was not constructed"
        assert (
            kwargs_list[0].get("managed_identity_client_id")
            == "11111111-2222-3333-4444-555555555555"
        )

    def test_no_client_id_omits_kwarg(self, fake_azure_identity):
        from agent.azure_identity_adapter import (
            EntraIdentityConfig,
            build_credential,
        )

        cfg = EntraIdentityConfig()
        build_credential(cfg)

        kwargs_list = fake_azure_identity["credential_kwargs"]
        assert kwargs_list
        assert "managed_identity_client_id" not in kwargs_list[0]

    def test_from_dict_picks_up_client_id(self):
        from agent.azure_identity_adapter import EntraIdentityConfig

        cfg = EntraIdentityConfig.from_dict({"client_id": "abc"})
        assert cfg.client_id == "abc"


# ---------------------------------------------------------------------------
# Cycle 2 — Runtime resolver accepts explicit ``auth_mode`` override
# (per-fallback entry config must defeat the top-level config)
# ---------------------------------------------------------------------------


class TestRuntimeResolverExplicitAuthMode:
    def _make_top_level_cfg(self):
        # Top-level model cfg uses api_key. The fallback entry says entra_id.
        return {
            "provider": "azure-foundry",
            "base_url": "https://r.openai.azure.com/openai/v1",
            "auth_mode": "api_key",
            "api_mode": "chat_completions",
        }

    def test_explicit_auth_mode_entra_overrides_config(self, fake_azure_identity):
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime

        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg=self._make_top_level_cfg(),
            explicit_base_url="https://foo.cognitiveservices.azure.com",
            explicit_auth_mode="entra_id",
            target_model="gpt-4o",
        )

        assert runtime["auth_mode"] == "entra_id"
        # Token provider — callable returning the fake bearer.
        assert callable(runtime["api_key"])
        assert runtime["api_key"]() == "fake-jwt"

    def test_explicit_client_id_passed_through_to_credential(self, fake_azure_identity):
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime

        _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg=self._make_top_level_cfg(),
            explicit_base_url="https://foo.cognitiveservices.azure.com",
            explicit_auth_mode="entra_id",
            explicit_client_id="user-assigned-mi-id",
            target_model="gpt-4o",
        )

        kwargs_list = fake_azure_identity["credential_kwargs"]
        assert kwargs_list
        assert kwargs_list[0].get("managed_identity_client_id") == "user-assigned-mi-id"

    def test_explicit_api_version_default_2024_10_21(self, fake_azure_identity):
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime

        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg=self._make_top_level_cfg(),
            explicit_base_url="https://foo.cognitiveservices.azure.com",
            explicit_auth_mode="entra_id",
            target_model="gpt-4o",
        )
        assert runtime.get("api_version") == "2024-10-21"

    def test_explicit_api_version_override(self, fake_azure_identity):
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime

        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg=self._make_top_level_cfg(),
            explicit_base_url="https://foo.cognitiveservices.azure.com",
            explicit_auth_mode="entra_id",
            explicit_api_version="2025-04-01-preview",
            target_model="gpt-4o",
        )
        assert runtime.get("api_version") == "2025-04-01-preview"


# ---------------------------------------------------------------------------
# Cycle 3 — Cognitiveservices base URL defaults the Entra scope to
# ``https://cognitiveservices.azure.com/.default``
# ---------------------------------------------------------------------------


class TestCognitiveServicesScope:
    def test_cognitiveservices_host_defaults_to_cognitive_scope(
        self, fake_azure_identity,
    ):
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime

        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg={
                "provider": "azure-foundry",
                "base_url": "https://x.cognitiveservices.azure.com",
                "auth_mode": "entra_id",
            },
            target_model="gpt-4o",
        )
        # Force the token provider to mint, which records the scope used.
        runtime["api_key"]()
        assert fake_azure_identity["scope"] == (
            "https://cognitiveservices.azure.com/.default"
        )
        assert fake_azure_identity["get_token_scopes"] == [
            "https://cognitiveservices.azure.com/.default",
        ]

    def test_cognitiveservices_host_with_trailing_slash_still_detected(
        self, fake_azure_identity,
    ):
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime

        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg={
                "provider": "azure-foundry",
                "base_url": "https://x.cognitiveservices.azure.com/",
                "auth_mode": "entra_id",
            },
            target_model="gpt-4o",
        )
        runtime["api_key"]()
        assert fake_azure_identity["scope"] == (
            "https://cognitiveservices.azure.com/.default"
        )

    def test_foundry_projects_host_keeps_existing_ai_azure_scope(
        self, fake_azure_identity,
    ):
        """Regression — ``r.openai.azure.com`` (Foundry Projects) still
        uses the legacy ``https://ai.azure.com/.default`` scope so
        existing entra_id deployments do not break."""
        from hermes_cli.runtime_provider import _resolve_azure_foundry_runtime

        runtime = _resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg={
                "provider": "azure-foundry",
                "base_url": "https://r.openai.azure.com/openai/v1",
                "auth_mode": "entra_id",
            },
            target_model="gpt-4o",
        )
        runtime["api_key"]()
        assert fake_azure_identity["scope"] == "https://ai.azure.com/.default"


# ---------------------------------------------------------------------------
# Cycle 4 — ``_try_azure_foundry`` builds an ``AzureOpenAI`` client (which
# constructs the AOAI deployment path with api-version) when the base URL
# is a bare ``cognitiveservices.azure.com`` host.
# ---------------------------------------------------------------------------


class TestAuxAzureFoundryAoaiClient:
    def test_cognitiveservices_url_uses_azure_openai_client_with_api_version(
        self, monkeypatch, fake_azure_identity,
    ):
        from agent import auxiliary_client as _aux

        received = {}

        class _FakeAzureOpenAI:
            def __init__(self, **kwargs):
                received.update(kwargs)
                self.api_key = kwargs.get("api_key") or kwargs.get(
                    "azure_ad_token_provider",
                )
                self.base_url = kwargs.get("azure_endpoint", "")

        monkeypatch.setattr(_aux, "AzureOpenAI", _FakeAzureOpenAI, raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"model": {
                "provider": "azure-foundry",
                "base_url": "https://x.cognitiveservices.azure.com",
                "auth_mode": "entra_id",
                "api_mode": "chat_completions",
                "default": "gpt-4o",
            }},
        )

        client, resolved = _aux._try_azure_foundry(model="gpt-4o")
        assert client is not None
        assert resolved == "gpt-4o"
        # AzureOpenAI constructor sees AOAI-specific kwargs.
        assert received.get("azure_endpoint") == (
            "https://x.cognitiveservices.azure.com"
        )
        assert received.get("api_version") == "2024-10-21"
        # The bearer-token callable arrived as the documented contract.
        provider = received.get("azure_ad_token_provider")
        assert callable(provider)
        assert provider() == "fake-jwt"

    def test_cognitiveservices_url_with_api_key_uses_azure_openai_too(
        self, monkeypatch,
    ):
        """api_key auth on a cognitiveservices URL still needs the AOAI
        deployment path — use AzureOpenAI with ``api_key=`` string."""
        from agent import auxiliary_client as _aux

        received = {}

        class _FakeAzureOpenAI:
            def __init__(self, **kwargs):
                received.update(kwargs)
                self.api_key = kwargs.get("api_key", "")
                self.base_url = kwargs.get("azure_endpoint", "")

        monkeypatch.setattr(_aux, "AzureOpenAI", _FakeAzureOpenAI, raising=False)
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "sk-static")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"model": {
                "provider": "azure-foundry",
                "base_url": "https://x.cognitiveservices.azure.com",
                "api_mode": "chat_completions",
                "default": "gpt-4o",
            }},
        )

        client, resolved = _aux._try_azure_foundry(model="gpt-4o")
        assert client is not None
        assert received.get("azure_endpoint") == "https://x.cognitiveservices.azure.com"
        assert received.get("api_version") == "2024-10-21"
        assert received.get("api_key") == "sk-static"


# ---------------------------------------------------------------------------
# Cycle 5 — Token / credential caching: second build_credential call with
# the same config returns the cached instance (so a token already minted
# by the bearer provider is reused inside azure-identity's own cache).
# ---------------------------------------------------------------------------


class TestCredentialCaching:
    def test_same_config_returns_cached_credential(self, fake_azure_identity):
        from agent.azure_identity_adapter import (
            EntraIdentityConfig,
            build_credential,
        )

        cfg = EntraIdentityConfig(scope="https://cognitiveservices.azure.com/.default")
        first = build_credential(cfg)
        second = build_credential(cfg)
        assert first is second
        # And only one DefaultAzureCredential constructed.
        assert len(fake_azure_identity["credential_kwargs"]) == 1

    def test_different_client_id_invalidates_cache(self, fake_azure_identity):
        """Distinct ``client_id`` values produce distinct credentials so
        ``managed_identity_client_id`` cannot leak across configs."""
        from agent.azure_identity_adapter import (
            EntraIdentityConfig,
            build_credential,
        )

        a = build_credential(EntraIdentityConfig(client_id="a"))
        b = build_credential(EntraIdentityConfig(client_id="b"))
        assert a is not b


# ---------------------------------------------------------------------------
# Cycle 6 — Fallback activation pipes the per-entry ``auth_mode`` /
# ``client_id`` / ``api_version`` through to ``resolve_provider_client``.
# Without this, a fallback entry's Entra config is silently dropped and
# the static-key path 401s against the cognitiveservices endpoint.
# ---------------------------------------------------------------------------


class TestFallbackForwardsAzureFields:
    def test_try_activate_fallback_forwards_auth_mode_and_client_id(
        self, monkeypatch,
    ):
        from agent import chat_completion_helpers as _h

        captured = {}

        def _fake_resolve(provider, **kwargs):
            captured["provider"] = provider
            captured.update(kwargs)
            # Return a minimal stand-in client so try_activate_fallback
            # proceeds far enough to verify the forward.
            return SimpleNamespace(
                api_key="bearer-callable-placeholder",
                base_url="https://x.cognitiveservices.azure.com",
            ), kwargs.get("model")

        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client", _fake_resolve,
        )
        monkeypatch.setattr(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            lambda m, p: m,
        )

        agent = SimpleNamespace(
            provider="zenops-shim",
            model="gpt-4o",
            base_url="http://127.0.0.1:8403",
            api_mode="chat_completions",
            api_key="local",
            client=SimpleNamespace(api_key="local", base_url="http://127.0.0.1:8403"),
            _fallback_chain=[{
                "provider": "azure-foundry",
                "model": "gpt-4o",
                "base_url": "https://x.cognitiveservices.azure.com",
                "auth_mode": "entra_id",
                "client_id": "uami-1234",
                "api_version": "2024-10-21",
            }],
            _fallback_index=0,
            _fallback_activated=False,
            _primary_runtime={"provider": "zenops-shim"},
            _rate_limited_until=0,
            _config_context_length=None,
            _transport_cache={},
            _credential_pool=None,
            _client_kwargs={},
            _is_azure_openai_url=lambda u: "cognitiveservices.azure.com" in u
                or "openai.azure.com" in u,
            _is_direct_openai_url=lambda u: False,
            _provider_model_requires_responses_api=lambda m, provider=None: False,
            _try_activate_fallback=lambda: False,
        )

        ok = _h.try_activate_fallback(agent)
        assert ok is True
        # The forward — these MUST reach resolve_provider_client for the
        # azure-foundry runtime resolver to honor the per-entry config.
        assert captured["provider"] == "azure-foundry"
        assert captured.get("auth_mode") == "entra_id"
        assert captured.get("client_id") == "uami-1234"
        assert captured.get("api_version") == "2024-10-21"
        assert captured.get("explicit_base_url") == (
            "https://x.cognitiveservices.azure.com"
        )
