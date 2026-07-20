import json
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from pydantic import BaseModel, Field, root_validator

class DataSource(BaseModel):
    type: str
    sql: Optional[str] = None

class Variable(BaseModel):
    name: str
    description: str
    type: str
    dataSource: Optional[DataSource] = None

class ProcessData(BaseModel):
    name: str
    type: str
    table: Optional[str] = None
    description: Optional[str] = None
    dataSource: Optional[DataSource] = None

class ProcessRole(BaseModel):
    name: str
    default: Optional[Any] = None
    endpoint: Optional[Any] = None
    resolutionRule: Optional[str] = None
    
class ProcessActivity(BaseModel):
    name: str
    id: str
    type: str
    description: str
    instruction: Optional[str] = None
    attachedEvents: Optional[List[str]] = Field(default_factory=list)
    role: str
    inputData: Optional[List[str]] = Field(default_factory=list)
    outputData: Optional[List[str]] = Field(default_factory=list)
    checkpoints: Optional[List[str]] = Field(default_factory=list)
    pythonCode: Optional[str] = None
    tool: Optional[str] = None
    properties: Optional[str] = None
    duration: Optional[int] = None
    srcTrg: Optional[str] = None
    agent: Optional[str] = None
    agentMode: Optional[str] = None
    orchestration: Optional[str] = None
    
    def __hash__(self):
        return hash(self.id)  # 또는 다른 고유한 속성을 사용

    def __eq__(self, other):
        if isinstance(other, ProcessActivity):
            return self.id == other.id  # 또는 다른 비교 로직을 사용
        return False

class SubProcess(BaseModel):
    name: str
    id: str
    type: str
    role: str
    attachedEvents: Optional[List[str]] = Field(default_factory=list)
    properties: Optional[str] = None
    duration: Optional[int] = None
    srcTrg: Optional[str] = None
    children: Optional["ProcessDefinition"] = None

class ProcessSequence(BaseModel):
    id: str
    name: Optional[str] = None
    source: str
    target: str
    condition: Optional[str] = None
    properties: Optional[str] = None

class ProcessGateway(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    role: Optional[str] = None
    type: Optional[str] = None
    process: Optional[str] = None
    condition: Optional[Dict[str, Any]] = Field(default_factory=dict)
    conditionData: Optional[List[str]] = None
    properties: Optional[str] = None
    description: Optional[str] = None
    srcTrg: Optional[str] = None
    duration: Optional[int] = None
    agentMode: Optional[str] = None
    orchestration: Optional[str] = None
    @root_validator(pre=True)
    def check_condition(cls, values):
        if values.get('condition') == "":
            values['condition'] = {}
        return values
class ProcessDefinition(BaseModel):
    processDefinitionName: str
    processDefinitionId: str
    description: Optional[str] = None
    data: Optional[List[ProcessData]] = []
    roles: Optional[List[ProcessRole]] = []
    activities: Optional[List[ProcessActivity]] = []
    subProcesses: Optional[List[SubProcess]] = []
    sequences: Optional[List[ProcessSequence]] = []
    gateways: Optional[List[ProcessGateway]] = []
    version_tag: Optional[str] = None
    version: Optional[str] = None

    def is_starting_activity(self, activity_id: str) -> bool:
        """
        Check if the given activity is the starting activity by verifying there's no previous activity.

        Args:
            activity_id (str): The ID of the activity to check.

        Returns:
            bool: True if it's the starting activity, False otherwise.
        """
        start_event = next((event for event in self.gateways if event.type == "startEvent"), None)
        if not start_event:
            return False

        for sequence in self.sequences:
            if sequence.source == start_event.id and sequence.target == activity_id:
                return True
        return False

    def find_initial_activity(self) -> Optional[ProcessActivity]:
        """
        Finds and returns the initial activity of the process, which is the one with no incoming sequences.

        Returns:
            Optional[Activity]: The initial activity if found, None otherwise.
        """
        start_event = next((event for event in self.gateways if event.type == "startEvent"), None)
        # Find the sequence with "start_event" as the source
        start_sequence = next((seq for seq in self.sequences if start_event.id in seq.source), None)
        
        if start_sequence:
            # Find the activity that matches the target of the start sequence
            return next((activity for activity in self.activities if activity.id == start_sequence.target), None)
        
        return None
    
    def find_prev_activity(self, current_activity_id: str) -> Optional[ProcessActivity]:
        for sequence in self.sequences:
            if sequence.target == current_activity_id:
                activity = self.find_activity_by_id(sequence.source)
                if activity:
                    return activity
                else:
                    gateway = self.find_gateway_by_id(sequence.source)
                    if gateway:
                        for sequence in self.sequences:
                            if sequence.target == gateway.id:
                                return self.find_prev_activity(sequence.source)
        return None
    
    def find_prev_activities(self, activity_id, prev_activities=None, visited=None) -> List[ProcessActivity]:
        if prev_activities is None:
            prev_activities = []
        if visited is None:
            visited = set()

        if activity_id in visited:
            return prev_activities

        visited.add(activity_id)

        # 현재 액티비티 또는 게이트웨이 찾기
        current = self.find_activity_by_id(activity_id)
        if current is None:
            current = self.find_gateway_by_id(activity_id)
            if current is None:
                return prev_activities

        # 현재 노드로 들어오는 모든 시퀀스 찾기
        incoming_sequences = [seq for seq in self.sequences if seq.target == activity_id]
        
        for sequence in incoming_sequences:
            source_id = sequence.source
            
            # 소스가 액티비티인 경우
            source_activity = self.find_activity_by_id(source_id)
            if source_activity and source_id not in visited:
                if source_activity not in prev_activities:
                    prev_activities.append(source_activity)
                self.find_prev_activities(source_id, prev_activities, visited)
                continue
            
            # 소스가 게이트웨이인 경우
            source_gateway = self.find_gateway_by_id(source_id)
            if source_gateway and source_id not in visited:
                self.find_prev_activities(source_id, prev_activities, visited)

        return prev_activities
    
    def find_next_item(self, current_item_id: str) -> Union[ProcessActivity, ProcessGateway]:
        for sequence in self.sequences:
            if sequence.source == current_item_id:
                source_id = sequence.target
                source_activity = self.find_activity_by_id(source_id)
                if source_activity:
                    return source_activity
                else:
                    source_gateway = self.find_gateway_by_id(source_id)
                    if source_gateway:
                        return source_gateway
        return None

    def find_next_activities(self, current_activity_id: str) -> List[ProcessActivity]:
        """
        Finds and returns the next activities in the process based on the current activity ID.

        Args:
            current_activity_id (str): The ID of the current activity.

        Returns:
            List[ProcessActivity]: A list of the next activities if found, empty list otherwise.
        """
        next_activities_ids = [sequence.target for sequence in self.sequences if sequence.source == current_activity_id]
        return [activity for activity in self.activities if activity.id in next_activities_ids]
    
    def find_end_activity(self) -> Optional[ProcessActivity]:
        """
        Finds and returns the end activity of the process, which is the one with no outgoing sequences.

        Returns:
            Optional[Activity]: The initial activity if found, None otherwise.
        """
        # Find the sequence with "end_event" as the source
        end_sequence = next((seq for seq in self.sequences if "end_event" in seq.target.lower()), None)
        
        if end_sequence:
            # Find the activity that matches the target of the start sequence
            return next((activity for activity in self.activities if activity.id == end_sequence.source), None)
        
        return None

    def find_activity_by_id(self, activity_id: str) -> Optional[ProcessActivity]:
        for activity in self.activities:
            if activity.id == activity_id:
                return activity
        return None
    
    def find_gateway_by_id(self, gateway_id: str) -> Optional[ProcessGateway]:
        for gateway in self.gateways:
            if gateway.id == gateway_id:
                return gateway
        return None

    def find_immediate_prev_activities(self, activity_id: str) -> List[ProcessActivity]:
        """
        현재 액티비티의 바로 이전 액티비티들을 찾습니다.
        게이트웨이를 통과하는 경우 게이트웨이 이전의 액티비티를 찾습니다.

        Args:
            activity_id (str): 현재 액티비티의 ID

        Returns:
            List[ProcessActivity]: 바로 이전 액티비티들의 목록
        """
        prev_activities = []
        visited = set()  # 순환 참조 방지를 위한 방문 체크
        
        def find_prev_through_gateway(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            
            # 현재 노드로 들어오는 시퀀스 찾기
            incoming = [seq for seq in self.sequences if seq.target == node_id]
            
            for seq in incoming:
                source_id = seq.source
                
                # 시작 이벤트는 건너뛰기
                if "start_event" in source_id.lower():
                    continue
                
                # 소스가 액티비티인 경우
                source_activity = self.find_activity_by_id(source_id)
                if source_activity:
                    if source_activity not in prev_activities:
                        prev_activities.append(source_activity)
                    continue
                
                # 소스가 게이트웨이인 경우
                source_gateway = self.find_gateway_by_id(source_id)
                if source_gateway:
                    # 게이트웨이로 들어오는 시퀀스 찾기
                    gateway_incoming = [seq for seq in self.sequences if seq.target == source_gateway.id]
                    for gw_seq in gateway_incoming:
                        gw_source = self.find_activity_by_id(gw_seq.source)
                        if gw_source and gw_source not in prev_activities:
                            prev_activities.append(gw_source)
        
        # 현재 액티비티로 들어오는 시퀀스 찾기
        current_incoming = [seq for seq in self.sequences if seq.target == activity_id]
        
        for sequence in current_incoming:
            source_id = sequence.source
            
            # 소스가 액티비티인 경우
            source_activity = self.find_activity_by_id(source_id)
            if source_activity:
                if source_activity not in prev_activities:
                    prev_activities.append(source_activity)
                continue
            
            # 소스가 게이트웨이인 경우
            source_gateway = self.find_gateway_by_id(source_id)
            if source_gateway:
                # 게이트웨이로 들어오는 시퀀스 찾기
                gateway_incoming = [seq for seq in self.sequences if seq.target == source_gateway.id]
                for gw_seq in gateway_incoming:
                    gw_source = self.find_activity_by_id(gw_seq.source)
                    if gw_source and gw_source not in prev_activities:
                        prev_activities.append(gw_source)
        
        return prev_activities
    
    def find_sequences(self, source_id: Optional[str], target_id: Optional[str]) -> List[ProcessSequence]:
        sequences = []
        for seq in self.sequences:
            if source_id is not None and seq.source == source_id:
                sequences.append(seq)
            if target_id is not None and seq.target == target_id:
                sequences.append(seq)
        return sequences
    
    def find_all_following_activities(self, activity_id: str, visited: Optional[set] = None) -> List[ProcessActivity]:
        """
        특정 액티비티 이후에 진행될 모든 액티비티 목록을 재귀적으로 추출합니다.
        
        Args:
            activity_id (str): 기준이 되는 액티비티 ID
            visited (Optional[set]): 순환 참조 방지를 위한 방문한 노드 집합
            
        Returns:
            List[ProcessActivity]: 해당 액티비티 이후에 진행될 모든 액티비티 목록
        """
        if visited is None:
            visited = set()
            
        # 순환 참조 방지
        if activity_id in visited:
            return []
            
        visited.add(activity_id)
        subsequent_activities = []
        
        # 현재 액티비티에서 나가는 모든 시퀀스 찾기
        outgoing_sequences = [seq for seq in self.sequences if seq.source == activity_id]
        
        for sequence in outgoing_sequences:
            target_id = sequence.target
            
            # 타겟이 액티비티인 경우
            target_activity = self.find_activity_by_id(target_id)
            if target_activity:
                if target_activity not in subsequent_activities:
                    subsequent_activities.append(target_activity)
                # 재귀적으로 해당 액티비티 이후의 모든 액티비티 찾기
                subsequent_activities.extend(self.find_all_following_activities(target_id, visited.copy()))
                continue
            
            # 타겟이 게이트웨이인 경우
            target_gateway = self.find_gateway_by_id(target_id)
            if target_gateway:
                # 게이트웨이에서 나가는 모든 시퀀스 찾기
                gateway_outgoing = [seq for seq in self.sequences if seq.source == target_gateway.id]
                for gw_seq in gateway_outgoing:
                    gw_target_activity = self.find_activity_by_id(gw_seq.target)
                    if gw_target_activity:
                        if gw_target_activity not in subsequent_activities:
                            subsequent_activities.append(gw_target_activity)
                        # 재귀적으로 해당 액티비티 이후의 모든 액티비티 찾기
                        subsequent_activities.extend(self.find_all_following_activities(gw_seq.target, visited.copy()))
        
        # 중복 제거
        unique_activities = []
        seen_ids = set()
        for activity in subsequent_activities:
            if activity.id not in seen_ids:
                unique_activities.append(activity)
                seen_ids.add(activity.id)
                
        return unique_activities

def load_process_definition(definition_json: dict) -> ProcessDefinition:
    # 입력 방어: DB 조회 실패 등으로 None/문자열이 들어오는 케이스를 안전하게 처리
    if definition_json is None:
        raise ValueError("Process definition JSON is None")
    if isinstance(definition_json, str):
        definition_json = json.loads(definition_json)
    if not isinstance(definition_json, dict):
        raise TypeError(f"Process definition JSON must be dict, got {type(definition_json).__name__}")

    # Events를 게이트웨이 리스트에 추가
    if 'events' in definition_json:
        if 'gateways' not in definition_json:
            definition_json['gateways'] = []
        for event in definition_json['events']:
            gateway = {
                'id': event['id'],
                'name': event.get('name', ''),
                'role': event.get('role', ''),
                'type': event['type'],
                'process': event.get('process', ''),
                'condition': event.get('condition', {}),
                'properties': event.get('properties', '{}'),
                'description': event.get('description', ''),
                'srcTrg': None
            }
            definition_json['gateways'].append(gateway)

    process_def = ProcessDefinition(**definition_json)
    
    # srcTrg 설정
    for sequence in process_def.sequences:
        # 타겟 액티비티 찾기
        target_activity = next((activity for activity in process_def.activities if activity.id == sequence.target), None)
        if target_activity:
            target_activity.srcTrg = sequence.source
            continue
            
        # 타겟 게이트웨이 찾기
        target_gateway = next((gateway for gateway in process_def.gateways if gateway.id == sequence.target), None)
        if target_gateway:
            target_gateway.srcTrg = sequence.source
            
    return process_def


# Stored gateway.type for events has been seen as both "startEvent"/"endEvent"
# (lowercase-first, e.g. tests/test.json) and "StartEvent"/"EndEvent" (schema example);
# match case-insensitively and normalize output to the schema's casing.
_EVENT_GATEWAY_TYPE_NAMES = {"startevent": "StartEvent", "endevent": "EndEvent"}


def _normalized_event_type(gateway_type: Optional[str]) -> Optional[str]:
    if not gateway_type:
        return None
    return _EVENT_GATEWAY_TYPE_NAMES.get(gateway_type.lower())


def _raw_activity_element(activity: ProcessActivity) -> dict:
    return {
        "elementType": "Activity",
        "id": activity.id,
        "name": activity.name,
        "type": activity.type,
        "source": activity.srcTrg or "",
        "description": activity.description,
        "instruction": activity.instruction,
        "role": activity.role,
        "skills": [],
        "tool": activity.tool,
        "agent": activity.agent,
        "agentMode": activity.agentMode,
        "orchestration": activity.orchestration,
        "inputData": activity.inputData or [],
        "outputData": activity.outputData or [],
        "checkpoints": activity.checkpoints or [],
        "duration": str(activity.duration) if activity.duration is not None else None,
    }


def _raw_sequence_element(sequence: ProcessSequence) -> dict:
    element = {
        "elementType": "Sequence",
        "id": sequence.id,
        "source": sequence.source,
        "target": sequence.target,
    }
    if sequence.name is not None:
        element["name"] = sequence.name
    if sequence.condition is not None:
        element["condition"] = sequence.condition
    return element


def _raw_gateway_or_event_element(gateway: ProcessGateway) -> dict:
    normalized_type = _normalized_event_type(gateway.type)
    if normalized_type:
        return {
            "elementType": "Event",
            "id": gateway.id,
            "name": gateway.name,
            "role": gateway.role,
            "source": gateway.srcTrg or "",
            "type": normalized_type,
            "description": gateway.description,
        }
    return {
        "elementType": "Gateway",
        "id": gateway.id,
        "name": gateway.name,
        "role": gateway.role,
        "source": gateway.srcTrg or "",
        "type": gateway.type,
        "description": gateway.description,
        "conditionData": gateway.conditionData or [],
    }


def _raw_role(role: ProcessRole) -> dict:
    # `origin` has no equivalent stored field on ProcessRole
    return {
        "name": role.name,
        "endpoint": role.endpoint,
        "resolutionRule": role.resolutionRule,
        "origin": None,
    }


def _raw_data(data: ProcessData) -> dict:
    return {
        "name": data.name,
        "description": data.description,
        "type": data.type,
    }


def _raw_elements(process_definition: "ProcessDefinition") -> List[dict]:
    elements = [_raw_activity_element(a) for a in (process_definition.activities or [])]
    elements += [_raw_sequence_element(s) for s in (process_definition.sequences or [])]
    elements += [_raw_gateway_or_event_element(g) for g in (process_definition.gateways or [])]
    return elements


def _raw_subprocess(sub_process: SubProcess) -> dict:
    children = sub_process.children
    raw_subprocess = {
        "id": sub_process.id,
        "name": sub_process.name,
        "role": sub_process.role,
        "type": sub_process.type,
        "process": children.processDefinitionId if children else None,
        "duration": str(sub_process.duration) if sub_process.duration is not None else None,
        "properties": sub_process.properties,
        "attachedEvents": sub_process.attachedEvents or [],
        "processDefinitionId": children.processDefinitionId if children else None,
        "processDefinitionName": children.processDefinitionName if children else None,
        "children": None,
    }
    if children:
        non_event_gateways = [g for g in (children.gateways or []) if not _normalized_event_type(g.type)]
        raw_subprocess["children"] = {
            "data": [_raw_data(d) for d in (children.data or [])],
            "roles": [_raw_role(r) for r in (children.roles or [])],
            # `ProcessDefinition` has no `events` field, so nested events can't be recovered
            "events": [],
            "gateways": [_raw_gateway_or_event_element(g) for g in non_event_gateways],
            "sequences": [_raw_sequence_element(s) for s in (children.sequences or [])],
            "activities": [_raw_activity_element(a) for a in (children.activities or [])],
            "subProcesses": [_raw_subprocess(sp) for sp in (children.subProcesses or [])],
        }
    return raw_subprocess


def convert_definition_to_raw_json(process_definition: "ProcessDefinition") -> dict:
    """
    Converts the stored ProcessDefinition shape into the raw process definition JSON
    shape (see process-definition.schema.json): a flat `elements` array discriminated
    by `elementType`, used as LLM input/output for the feedback-diff endpoint.
    `isHorizontal`, `Activity.skills`, and `Role.origin` have no equivalent stored
    field and are always emitted as a default/empty.
    """
    return {
        "processDefinitionName": process_definition.processDefinitionName,
        "processDefinitionId": process_definition.processDefinitionId,
        "description": process_definition.description,
        "isHorizontal": True,
        "data": [_raw_data(d) for d in (process_definition.data or [])],
        "roles": [_raw_role(r) for r in (process_definition.roles or [])],
        "elements": _raw_elements(process_definition),
        "subProcesses": [_raw_subprocess(sp) for sp in (process_definition.subProcesses or [])],
    }


# Example usage
if __name__ == "__main__":
    json_str = '{"processDefinitionName": "Example Process", "processDefinitionId": "example_process", "description": "제 프로세스 설명", "data": [{"name": "example data", "description": "example data description", "type": "Text"}], "roles": [{"name": "example role", "resolutionRule": "example rule"}], "activities": [{"name": "example activity", "id": "example_activity", "type": "ScriptActivity", "description": "activity description", "instruction": "activity instruction", "role": "example role", "inputData": [{"name": "example input data"}], "outputData": [{"name": "example output data"}], "checkpoints":["checkpoint 1"], "pythonCode": "import smtplib\\nfrom email.mime.multipart import MIMEMultipart\\nfrom email.mime.text import MIMEText\\n\\nsmtp = smtplib.SMTP(\'smtp.gmail.com\', 587)\\nsmtp.starttls()\\nsmtp.login(\'jinyoungj@gmail.com\', \'raqw nmmn xuuc bsyi\')\\n\\nmsg = MIMEMultipart()\\nmsg[\'Subject\'] = \'Test mail\'\\nmsg.attach(MIMEText(\'This is a test mail.\'))\\n\\nsmtp.sendmail(\'jinyoungj@gmail.com\', \'ohsy818@gmail.com\', msg.as_string())\\nsmtp.quit()"}], "sequences": [{"source": "activity_id_1", "target": "activity_id_2"}]}'
    process_definition = load_process_definition(json_str)
    print(process_definition.processDefinitionName)

    current_dir = Path(__file__).parent

    # from code_executor import execute_python_code

    for activity in process_definition.activities:
        if activity.type == "ScriptActivity":
            print(activity)
            # execute_python_code(activity.pythonCode, current_dir)
            # output = execute_python_code(activity.pythonCode, current_dir)
            # print(output)
    # End Generation Here

class UIDefinition(BaseModel):
    id: str
    html: str
    proc_def_id: Optional[str] = None
    activity_id: Optional[str] = None
    fields_json: Optional[List[Dict[str, Any]]] = None