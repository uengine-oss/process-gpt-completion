# process-gpt-completion

## Environment Setup

Before running the project, you need to set up environment variables.

```
# Copy .env.example to create .env file
cp .env.example .env

# Edit .env file with your actual values
# - OPENAI_API_KEY: Your OpenAI API key
# - SUPABASE_URL, SUPABASE_KEY: Your Supabase configuration. SUPABASE_KEY should be a Service Role key
# - Other required environment variables
```

## Install Dev Env (using uv)
```
supabase start

uv venv .venv
uv pip install -r requirements.txt

# Mac/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

uv run main.py
```

## Polling Service  (using uv)

The polling service handles workitem polling and processing. See `polling_service/README.md` for detailed setup and deployment instructions.

```
cd polling_service
uv venv .venv
uv pip install -r requirements.txt

# Mac/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

uv run main.py
```

## Supabase 접속
```
http://localhost:54323
```

# API Test

```
http POST localhost:8000/complete \
  Content-Type:application/json \
  input:='{
    "process_definition_id": "contest_submission_evaluation",
    "process_instance_id": "new",
    "email": "help@uengine.org",
    "role_mappings": [
      {
        "name": "참가자",
        "endpoint": "help@uengine.org",
        "resolutionRule": "공모전 참가자 이메일로 식별"
      },
      {
        "name": "평가담당자",
        "endpoint": "help@uengine.org",
        "resolutionRule": "프로세스 내 지정"
      }
    ],
    "answer": "새로운 아이디어를 공모전에 제출하겠습니다. 아이디어명: AI 기반 업무 자동화 솔루션, 제출자: 홍길동, 설명: 인공지능을 활용하여 반복적인 업무를 자동화하는 혁신적인 솔루션입니다.",
    "chat_room_id": "contest_submission_evaluation.uuid-1234-5678",
    "form_values": {
      "customer_email": "help@uengine.org",
      "idea_name": "AI 기반 업무 자동화 솔루션",
      "submitter_name": "홍길동",
      "idea_description": "인공지능을 활용하여 반복적인 업무를 자동화하고 효율성을 극대화하는 혁신적인 솔루션입니다. 기존 수작업으로 처리되던 업무들을 AI가 학습하고 자동으로 처리할 수 있도록 지원합니다."
    }
  }'
```

## Old API

```
INST_ID=$(http :8000/complete/invoke input[process_instance_id]="new" input[process_definition_id]="company_entrance" | python3 -c "import sys, json; print(json.loads(json.loads(sys.stdin.read())['output'])['instanceId'])")
echo $INST_ID
http :8000/complete/invoke input[answer]="지원분야는 SW engineer" input[process_instance_id]="$INST_ID" input[activity_id]="congrate" # 400  error
http :8000/complete/invoke input[answer]="지원분야는 SW engineer" input[process_instance_id]="invalid instance id" input[activity_id]="registration"  # 404 error
http :8000/complete/invoke input[answer]="지원분야는 SW engineer" input[process_instance_id]="$INST_ID" input[activity_id]="registration"  | python3 -c "import sys, json; print(json.loads(json.loads(sys.stdin.read())['output'])['nextActivities'])" 
# next activity id should be 'nextMail'
http :8000/complete/invoke input[answer]="no comment" input[process_instance_id]="$INST_ID" input[activity_id]="nextMail"


# 입사지원2: 입사 지원서 이미지 파일을 기반으로한: 
INST_ID=$(http :8000/complete/invoke input[process_instance_id]="new" input[process_definition_id]="company_entrance" | python3 -c "import sys, json; print(json.loads(json.loads(sys.stdin.read())['output'])['instanceId'])")
echo $INST_ID

http :8000/vision-complete/invoke input[answer]="세부 지원사항은 지원서에 확인해주십시오" input[process_instance_id]="$INST_ID" input[activity_id]="registration" 


# vacation use process
INST_ID=$(http :8000/complete/invoke input[process_instance_id]="new" input[process_definition_id]="vacation_request" input[answer]="The total number of vacation days requested is 5, starting from February 5, 2024, to February 10, 2024, for the reason of travel" | python3 -c "import sys, json; print(json.loads(json.loads(sys.stdin.read())['output'])['instanceId'])")
echo $INST_ID
http :8000/complete/invoke input[answer]="승인합니다" input[process_instance_id]="$INST_ID" input[activity_id]="manager_approval" # 400  error

# vacation addition process
INST_ID=$(http :8000/complete/invoke input[process_instance_id]="new" input[process_definition_id]="vacation_addition" input[answer]="5일간 휴가를 추가합니다" | python3 -c "import sys, json; print(json.loads(json.loads(sys.stdin.read())['output'])['instanceId'])")
echo $INST_ID
http :8000/complete/invoke input[answer]="승인합니다" input[process_instance_id]="$INST_ID" input[activity_id]="manager_approval" # 400  error
```



## Setting VSCode Debug Env
- Press Cmd+P to open search box and Enter ">" to select ">Python Interpreter" and set the right environment.
- Open main.py and Set breakpoints on main.py at some line to stop the execution:

```
def combine_input_with_process_definition(input):
    # 프로세스 인스턴스를 DB에서 검색
    
    process_instance_id = input.get('process_instance_id')  # 'process_instance_id' 키에 대한 접근 추가
>>    activity_id = input.get('activity_id')  <<< DEBUG POINT

```

- Switch to the Debug perspective 
- Press the Run button with the option "Python: Current File" (don't forget to leave the main.py file opened and the editor tab selected)

- You may face some dependency error or OPENAI_API_KEY related error. When it comes to you, do these:
```
# firstly Ctrl+C to stop the current debug session inside the terminal the debugger runs
# insert these commands:
pipenv shell
export OPENAI_API_KEY=sk-...
# Try to run the debugger again (Cmd+Shift+D to swith to the Debug perspective and just press enter)
```

