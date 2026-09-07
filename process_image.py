from fastapi import HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from llm_factory import create_llm

class ImageData(BaseModel):
    image_base64: str

async def process_image(data: ImageData):
    try:
        client = create_llm(
            streaming=False,
            temperature=0,
            max_tokens=4096,
        )
        response = await client.ainvoke(
            [
                SystemMessage(content="""
너는 문서 편집장이야. 그렇게 깐깐하 사람은 아니고 비교적 문서 편집상태가 불량하더라도 칭찬을 하는 편이고 그냥 넘어가는 스타일이야. 제출한 문서의 폰트가 일정한지, 들여쓰기 내쓰기가 올바른지, 번호매김 서식 등이 일정한지, 장평이 일정한지, 글의 흐름이 정련한지, 개조식인지 서술식인지에 따라 일관성이 있는지, 일관된 문체인지 (구어체인지 문어체인지) 등을 평가해줘. 비교적 잘된 문서라면 잘했다고 긍정적으로 평가하고, 너무 잘못된 문서에 대해서만 지적해줘. 지적 사항 중 가장 심각한 순서대로하여 3개를 넘지 않도록 해줘.     
                """),
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{data.image_base64}",
                                'detail': 'high'
                            }
                        }, 
                        {"type": "text", "text": "\n이 문서에 잘못된 점을 지적해주세요.\n"},
                    ],
                ),
            ]
        )
        return {"message": getattr(response, "content", str(response))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def add_routes_to_app(app):
    app.add_api_route("/process-image/", process_image, methods=["POST"])
