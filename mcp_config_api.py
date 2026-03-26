from fastapi import HTTPException
import copy
import json
from typing import Dict

def add_routes_to_app(app):
    app.add_api_route("/mcp-tools", load_mcp_tools, methods=["GET"])


def _sanitize_mcp_servers(config: Dict) -> Dict:
    sanitized = copy.deepcopy(config)

    for server in sanitized.values():
        env = server.get("env")
        if isinstance(env, dict):
            server["env"] = {key: "***" for key in env}

    return sanitized


def load_mcp_tools() -> Dict:
    """Load and return MCP configuration from mcp.json file."""
    try:
        with open('mcp.json', 'r') as f:
            mcp_config = json.load(f)
            return _sanitize_mcp_servers(mcp_config.get("mcpServers", {}))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=404, detail=f"Failed to load MCP config: {str(e)}")

    
