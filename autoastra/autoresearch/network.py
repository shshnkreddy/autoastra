"""Provider endpoints allowed by default; arbitrary internet remains blocked."""


def model_hosts(config):
    provider = config.provider
    if provider == 'codex':
        return ('api.openai.com', 'chatgpt.com', 'auth.openai.com')
    if provider == 'opencode':
        provider = config.model.partition('/')[0]
    hosts = {
        'openai': ('api.openai.com',),
        'openrouter': ('openrouter.ai',),
        'anthropic': ('api.anthropic.com', 'platform.claude.com', 'console.anthropic.com', 'claude.ai'),
        'opencode': ('opencode.ai',),
        'opencode-go': ('opencode.ai',),
    }
    if provider not in hosts:
        raise ValueError(f'No model-domain policy for {provider!r}; supply SandboxConfig.model_hosts explicitly')
    return hosts[provider]
