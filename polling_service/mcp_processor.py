from typing import Dict, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
from database import fetch_tenant_mcp_config, fetch_process_definition_by_version, fetch_workitem_by_proc_inst_and_activity, fetch_process_instance
from process_definition import load_process_definition
from langchain.tools import StructuredTool
from pydantic import BaseModel
from langchain.tools import StructuredTool
from llm_factory import create_llm

if os.getenv("ENV") != "production":
    load_dotenv(override=True)

class EmptySchema(BaseModel):
    """빈 파라미터용 스키마"""
    pass

def sanitize_mcp_tools(tools):
    sanitized = []
    for tool in tools:
        # dict 형태로 들어온 args_schema는 잘못된 가능성 높음
        if isinstance(tool.args_schema, dict):
            schema_dict = tool.args_schema
            has_properties = schema_dict.get('properties')
            has_required = schema_dict.get('required')

            if not has_properties and not has_required:
                print(f"[WARNING] Tool {tool.name} has invalid schema. Patching with EmptySchema.")
                try:
                    patched_tool = StructuredTool.from_function(
                        func=tool.coroutine,
                        name=tool.name,
                        description=tool.description,
                        args_schema=EmptySchema,
                        coroutine=tool.coroutine
                    )
                    sanitized.append(patched_tool)
                    continue
                except Exception as e:
                    print(f"[ERROR] Failed to patch tool {tool.name}: {e}")
                    continue
        else:
            # 정상적인 StructuredTool은 그대로 추가
            sanitized.append(tool)
    return sanitized

class MCPProcessor:
    def __init__(self):
        self.mcp_client = None
        self.mcp_tools = {}
        self.mcp_configs = {}
    
    async def get_mcp_tools_from_tenant(self, tenant_id: str, agent_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        테넌트의 mcp 설정에서 에이전트의 tools를 확인하고 가져옵니다.
        """
        try:
            mcp_config = fetch_tenant_mcp_config(tenant_id)
            if not mcp_config:
                print(f"[WARNING] No MCP config found for tenant: {tenant_id}")
                return {}
            
            mcp_servers = mcp_config.get("mcpServers", {})
            
            agent_tools = agent_info.get("tools", '')
            if not agent_tools:
                print(f"[WARNING] No tools found for agent: {agent_info.get('id')}")
                return {}
            
            if ',' in agent_tools:
                agent_tools = agent_tools.split(',')
            else:
                agent_tools = [agent_tools]
            
            available_tools = {}
            for tool_name in agent_tools:
                if tool_name in mcp_servers:
                    tool_config = mcp_servers[tool_name]
                    available_tools[tool_name] = tool_config
                    print(f"[INFO] Found MCP tool: {tool_name} for agent: {agent_info.get('id')}")
                else:
                    print(f"[WARNING] Tool {tool_name} not found in MCP config for tenant: {tenant_id}")
            
            return available_tools
            
        except Exception as e:
            print(f"[ERROR] Failed to get MCP tools from tenant: {str(e)}")
            return {}
    
    async def initialize_mcp_client(self, tenant_id: str, agent_info: Dict[str, Any]) -> bool:
        """
        MultiServerMCPClient를 초기화하고 MCP 도구들을 로드합니다.
        """
        try:
            available_tools = await self.get_mcp_tools_from_tenant(tenant_id, agent_info)
            if not available_tools:
                return False
            
            for tool_name, tool_config in available_tools.items():
                self.mcp_configs[tool_name] = tool_config
            
            # MultiServerMCPClient 초기화
            self.mcp_client = MultiServerMCPClient(self.mcp_configs)
            
            # MCP 도구들 로드
            self.mcp_tools = await self.mcp_client.get_tools()

            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize MCP client: {str(e)}")
            return False
    
    def wrap_tool_for_empty_args(self, tool):
        orig_coroutine = tool.coroutine
        async def wrapped_coroutine(*args, **kwargs):
            # 만약 파라미터가 없거나 None이면 빈 dict로 대체
            if not args and not kwargs:
                return await orig_coroutine({})
            return await orig_coroutine(*args, **kwargs)
        tool.coroutine = wrapped_coroutine
        return tool

    async def execute_mcp_tools(self, workitem: Dict[str, Any], agent_info: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        """
        MCP 도구들을 사용하여 서비스 업무를 실행합니다.
        """
        try:
            tools = agent_info.get('tools', '')
            if not tools:
                return {"error": "No tools found for agent"}

            if ',' in tools:
                tools = tools.split(',')
            else:
                tools = [tools]

            await self.initialize_mcp_client(tenant_id, agent_info)

            proc_def_id = workitem.get('proc_def_id')
            version_tag = workitem.get('version_tag')
            version = workitem.get('version')
            arcv_id = None
            if not version_tag and not version and workitem.get('proc_inst_id'):
                try:
                    process_instance = fetch_process_instance(workitem.get('proc_inst_id'), tenant_id)
                    if process_instance and getattr(process_instance, "proc_def_version", None):
                        arcv_id = process_instance.proc_def_version
                except Exception as e:
                    print(f"[WARN] Failed to fetch process instance for MCP version: {e}")

            process_definition_json = fetch_process_definition_by_version(
                proc_def_id,
                version_tag,
                version,
                tenant_id,
                arcv_id,
            )
            process_definition = load_process_definition(process_definition_json)
            activity = process_definition.find_activity_by_id(workitem.get('activity_id'))
            if not activity:
                return {"error": "Activity not found"}
            
            activity_description = activity.description
            activity_name = activity.name
            prev_activities = process_definition.find_prev_activities(activity.id, [])
            prev_activities_output = ""
            
            if prev_activities:
                for prev_activity in prev_activities:
                    prev_workitem = fetch_workitem_by_proc_inst_and_activity(workitem.get('proc_inst_id'), prev_activity.id, tenant_id)
                    prev_activities_output += f"{prev_activity.name}: {prev_workitem.output}\n"
            else:
                prev_activities_output = "이전 산출물이 없습니다."

            prompt = f"""
활동 이름: {activity_name}
활동 설명: {activity_description}

위 활동을 수행하기 위해 사용 가능한 도구들을 활용해주세요.
도구를 사용할 때는 각 도구가 요구하는 파라미터 스키마에 맞게 파라미터를 전달해야 합니다.
파라미터가 없는 도구는 빈 오브젝트({{}})를 전달하세요.

이전 산출물: {prev_activities_output}
"""

            tools = sanitize_mcp_tools(self.mcp_tools)
            # tools = [self.wrap_tool_for_empty_args(tool) for tool in tools]
            llm = create_llm(streaming=False)
            agent = create_react_agent(llm, tools)
            response = await agent.ainvoke({"messages": prompt})
            
            return response
            
        except Exception as e:
            print(f"[ERROR] Failed to execute MCP tools: {str(e)}")
            return {"error": str(e)}

    async def cleanup(self):
        """
        리소스 정리 (MCP 클라이언트 및 도구들 정리)
        """
        try:
            # MCP 클라이언트 정리
            if self.mcp_client:
                try:
                    await self.mcp_client.close()
                    print("[INFO] Closed MCP client")
                except Exception as e:
                    print(f"[ERROR] Failed to close MCP client: {str(e)}")
            
            # 도구 정보 초기화
            self.mcp_tools.clear()
            self.mcp_configs.clear()
            self.mcp_client = None
            print("[INFO] MCP client and tools cleaned up")
            
        except Exception as e:
            print(f"[ERROR] Failed to cleanup MCP client: {str(e)}")

# 전역 MCP 프로세서 인스턴스
mcp_processor = MCPProcessor()
