from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
import os

os.environ["OPENAI_API_KEY"] = os.getenv('PUBLIC_SERVICE_KEY')

llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo-0613')

def response(message, history):
    history_langchain_format = []    
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human)) # 사람의 메시지
        history_langchain_format.append(AIMessage(content=ai)) # AI의 응답
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm(history_langchain_format) # 사용자가 보낸 새로운 메시지
    return gpt_response.content

demo = gr.ChatInterface(
  fn=response,
  textbox=gr.Textbox(placeholder="말걸어주세요..", container=False, scale=7),
  # 채팅창의 크기를 조절한다.
  chatbot=gr.Chatbot(height=1000),
  title="ChatGPT를 이용한 챗봇",
  description="무엇이든 물어보세요!!",
  theme="soft",
  examples=["파이썬이란?", "프로그래밍 언어 학습법", "gradio란?"],
  retry_btn="다시보내기 ↩",
  undo_btn="이전챗 삭제 ❌",
  clear_btn="전챗 삭제 💫",  
  css="style.css",
)

if __name__ == "__main__":
    demo.launch()