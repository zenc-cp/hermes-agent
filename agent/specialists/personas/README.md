# Specialist Personas

Each specialist persona is defined by a YAML file that specifies:
- **name**: The specialist identifier (e.g., `Scout`, `Hunter`)
- **system_prompt**: Instructions for the specialist's behavior
- **allowed_tools**: List of tool names the specialist can use
- **default_model**: The model to use for this specialist
- **output_schema**: JSON Schema describing the expected output format

## Tool Vocabulary

Persona YAML files reference tools by their real Hermes names. The mapping between persona tool names and Hermes toolsets is:

| Tool Name | Toolset |
|-----------|---------|
| `web_search` | `web` |
| `web_extract` | `web` |
| `terminal` | `terminal` |
| `read_file` | `file` |
| `write_file` | `file` |

### Adding New Tools

When adding a new tool to a persona's `allowed_tools` list:

1. **The tool name MUST exist** in Hermes `toolsets.py:_HERMES_CORE_TOOLS`
2. **Update `_TOOL_TO_TOOLSET` mapping** in `agent/specialists/consumer.py` if the tool maps to a new toolset
3. The validator in `load_persona()` will reject unknown tools at load time

### Example

Valid `allowed_tools`:
```yaml
allowed_tools: [web_search, web_extract, read_file]
```

Invalid (will be rejected at load):
```yaml
allowed_tools: [web_search, fake_tool]  # Error: "fake_tool" not in _TOOL_TO_TOOLSET
```

## Validation

- `load_persona()` validates all tool names against the set of known Hermes tools
- Validation happens at load time; invalid personas fail with clear error messages
- No persona-specific tool conditionals (e.g., `if persona.name == "Scout"`) — all behavior is YAML-only
