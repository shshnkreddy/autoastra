"""Model access and session settings, separate from tasks and Docker resources."""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class RunnerConfig:
    model: str | None = None
    provider: str = 'codex'
    codex_binary: Path | None = None
    codex_home: Path | None = None
    startup_timeout_seconds: int = 60
    conclusion_seconds: int = 10
    claude_binary: Path | None = None
    opencode_binary: Path | None = None

    @property
    def profile_path(self):
        return Path(self.codex_home or os.environ.get(
            'AUTOASTRA_CODEX_HOME', '~/.local/share/autoastra/codex')).expanduser()

    @property
    def agent(self):
        if self.provider == 'opencode':
            return 'opencode'
        return 'claude' if self.provider == 'anthropic' else 'codex'

    def __post_init__(self):
        if self.provider not in ('codex', 'openrouter', 'anthropic', 'opencode'):
            raise ValueError('provider must be codex, openrouter, anthropic, or opencode')
        if self.agent != 'codex' and (self.codex_binary or self.codex_home):
            raise ValueError('Codex options require the codex or openrouter provider')
        if self.agent != 'claude' and self.claude_binary:
            raise ValueError('--claude-binary requires the anthropic provider')
        if self.agent != 'opencode' and self.opencode_binary:
            raise ValueError('--opencode-binary requires the opencode provider')
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError('model must be a nonempty name')
        if self.agent == 'opencode':
            provider, _, model = (self.model or '').partition('/')
            if not provider.strip() or not model.strip():
                raise ValueError('Choose an OpenCode model with --model provider/model')
        if self.provider == 'openrouter' and not self.model:
            raise ValueError('Choose an OpenRouter model with --model')
        for name in ('startup_timeout_seconds', 'conclusion_seconds'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
