# 基于NVIDIA NIM平台的本地边缘设备大模型部署

摘要：笔者在[NVIDIA DLI](https://www.nvidia.cn/training/)上报名了NVIDIA AI-AGENT夏季训练营，这个课程用三天让我们实现基于[NVIDIA NIM](https://build.nvidia.com/explore/discover)平台的RAG-AIAgent运用。

## 环境部署  

NVIDIA课程的环境部署在CSDN博客:[2024 NVIDIA开发者夏令营环境配置指南(Win&Mac)](https://blog.csdn.net/kunhe0512/article/details/140910139)中有详细说明。同时，本课程还邀请了Microsoft的工程师作为第二天内容的讲师，对于Microsoft课程内容的环境配置，在GitHub[Microsoft-Phi-3-NvidiaNIMWorkshop](https://github.com/kinfey/Microsoft-Phi-3-NvidiaNIMWorkshop?tab=readme-ov-file)。故笔者不再对个人的环境部署作详细说明。

## 项目概述

笔者基于课程所学的内容，采用了**LangChain**这个python包和**OpenAI**包提供的AI接口以及[NVIDIA](https://build.nvidia.com/explore/discover)的各式各样的LLM模型实现了一个RAG AIAgent，还进一步实现了图片输入的多模态LLM。

![image](https://raw.githubusercontent.com/awa2333/awa2333.github.io/main/img/img1.png "个人成果")

总体来说，这是一个最基本的成果，课程上还有更加惊艳的成果，但是笔者不再赘述。

## 技术方案与实施步骤

主要说明笔者个人对于llm和RAG以及他们实现方法的浅薄的理解。

### 模型选择

本项目在视觉识别上使用了[Microsoft Phi3](https://azure.microsoft.com/zh-cn/products/phi-3/)模型。在处理用户输入上，笔者采用了[meta llama3](https://github.com/meta-llama/llama3)模型，对于RAG模型，笔者使用了[NVIDIA Embed-QA](https://build.nvidia.com/nvidia/embed-qa-4)模型。当然对于选择什么模型比较符合个人的需求，笔者并不能给出一个具体的答复，毕竟这算是笔者首次接触到LLM，甚至于对于RAG究竟是什么，也是在课上方才了解一二。

### 数据构建

对于数据的构建与处理，说实话我依旧难以作出我自己的解答，不过大致应是基于用户输入对文本数据的综合检索，作为增强输入传递到llm再生成数据，也即RAG或者说检索增强生成。具体而言，RAG会把文本数据向量化，同时也把输入向量化，从文本数据的向量中找寻与输入的向量较为***相似***的向量。避免了llm在长文本输入中出现的幻觉现象。

### 功能整合

课程上第二天和第三天的安排是对于多模态大模型的实现方法作出简要介绍。笔者不才，未能实现多模态的RAG，只能寄希望于未来的某一天可以实现多模态RAG。笔者未能吃透第二天的学习内容，第三天的内容是***简单***多模态llm的实现。笔者的简单多模态llm是指额外多了一个图片输入，同时可以让llm直接运行自己生成的代码，而不必自己手动生成。

## 实施步骤

主要说明复现代码的方式。

### 环境搭建

环境搭建方面笔者使用的是Linux平台，Python3.8下所有的包都能直接使用pip安装，并没有踩坑。按照课程的学习，理应还应该补上*gradio*包和*langchain-community*包，无论Linux平台还是Linux平台，都可以使用`pip install gradio`和`pip install langchain-community`直接安装。

### 代码实现

因为代码较多，笔者只对较为重要的代码作出自己的理解。

#### main()

首先是主函数。`get_nvidia_key()`从字面上理解就是输入[NVIDIA NIM](https://build.nvidia.com/nvidia/embed-qa-4)平台的密钥，而model和embedder则是选择的模型。当然笔者的Python编程水平较为低下，难以实现面向对象编程。所以代码中的`model`和`embedder`并不能影响下文的模型，当然待笔者编程能力提升了，应该有办法实现的。而`model_initial()`则是对现有的文本向量化，只要模型和文本没有改变，可以不再执行。所以主函数中我把它注释了。对于用户UI，笔者这里使用*gradio*包实现。

```python
def main():
    get_nvidia_key()
    model = "meta/llama-3.1-405b-instruct"
    embedder = "NV-Embed-QA"
    # model_initial(model, embedder)
    gradio_interface = gradio.Interface(fn=model_process,
                                        inputs=[gradio.Image(label="Upload image", type="filepath"), 'text'],
                                        outputs=['image'],
                                        title="Multi Modal chat agent",
                                        description="Multi Modal chat agent",
                                        allow_flagging="never")
    gradio_interface.launch(debug=True, share=False, show_api=False)
    return None
```

#### model_process()

核心的处理自然是使用llm，这里我们首先是调用`microsoft/phi-3-vision-128k-instruct`作为输入图像处理的模型，接着以`meta/llama-3.1-405b-instruct`作为综合数据处理的llm。

```python
def model_process(image_b64, user_input, table):
    image_b64 = image2b64(image_b64)
    chart_reading = ChatNVIDIA(model="microsoft/phi-3-vision-128k-instruct")
    chart_reading_prompt = ChatPromptTemplate.from_template(
        'Generate underlying data table of the figure below, : <img src="data:image/png;base64,{image_b64}" />'
    )
    chart_chain = chart_reading_prompt | chart_reading
    instruct_chat = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
    instruct_prompt = ChatPromptTemplate.from_template(
        "Do NOT repeat my requirements already stated. Based on this table {table}, {input}" \
        "If has table string, start with 'TABLE', end with 'END_TABLE'." \
        "If has code, start with '```python' and end with '```'." \
        "Do NOT include table inside code, and vice versa."
    )
    instruct_chain = instruct_prompt | instruct_chat
    chart_reading_branch = RunnableBranch(
        (lambda x: x.get('table') is None, RunnableAssign({'table': chart_chain})),
        (lambda x: x.get('table') is not None, lambda x: x),
        lambda x: x
    )
    update_table = RunnableBranch(
        (lambda x: 'TABLE' in x.content, save_table_to_global),
        lambda x: x
    )
    execute_code = RunnableBranch(
        (lambda x: '```python' in x.content, execute_and_return_gr),
        lambda x: x
    )
    chain = (
            chart_reading_branch
            | instruct_chain
            | update_table
            | execute_code
    )
    return chain.invoke({"image_b64": image_b64, "input": user_input, "table": table})
```

## 项目成果与展示

笔者学习三天暑季训练营的成果展示

### 项目应用场景

当下，人工智能已经成为互联网新的风口。基于RAG的AIAgent理论上可以部分取代甚至全部取代当下的信息搜索如百度，服务咨询如客服等等的工作，释放生产力。

### 功能演示

上文已经展示了笔者手下的多模态AIAgent的成果，比方说可以生成代码和以图生图，笔者就不在这里过多赘述。

## 项目总结与展望

评估项目完成度与应用价值，以及未来的应用方向。

### 项目评估

显然，以短短三天的训练营，并且在这之前几乎没有任何的LLM方面的了解的情况下，作出一个高完成度的AIAgent是一件颇副挑战性的工作。不过笔者也还是尽量交出了自己的答卷。总体而言，因为笔者有一定的python编程经验，所以我有比其他少部分没有python经验的同学更快实现代码。但是不足也就在这里，对于LLM不了解，几乎改不动课程的代码，在别人都在讨论多模态RAG时，我依旧没能解决自己的报错。所以没能实现多模态的RAG AIAgent

### 未来方向

如上文所说，未来可以尝试实现真正的多模态RAG AIAgent。

## 代码

```python
import base64
from itertools import chain
from marshal import dumps
from os.path import split
import gradio
from IPython.core.debugger import prompt
from idna import valid_label_length
from langchain_community.vectorstores import FAISS
from openai import OpenAI
import getpass
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from sqlalchemy.testing.suite.test_reflection import metadata
from tqdm import tqdm
from pathlib import Path
from operator import itemgetter
from langchain.vectorstores import VectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
import faiss
from triton.language.semantic import store
from IPython.display import Audio
from langchain.load.dump import dumps
import matplotlib.pyplot as plt
import numpy as np
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable import RunnablePassthrough
import re
from langchain_core.runnables import RunnableBranch, RunnableAssign

global img_path
img_path = '/home/he/CodeField/Python/py10_microsoft_llm/workspace/Microsoft-Phi-3-NvidiaNIMWorkshop/' + 'image.png'


def get_nvidia_key():
    if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
        print("Valiad NVIDIA_API_KEY already in environment. Delete to reset")
    else:
        nvapi_key = getpass.getpass("Enter your NVIDIA API Key: ")
        assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid NVIDIA API Key"
        os.environ["NVIDIA_API_KEY"] = nvapi_key
        return None


def rag_initial(chat_model, embedder_model):
    return ChatNVIDIA(model=chat_model, max_tokens=2048), NVIDIAEmbeddings(model=embedder_model)


def save_table_to_global(x):
    global table
    if 'TABLE' in x.content:
        table = x.content.split('TABLE', 1)[1].split('END_TABLE')[0]
    return x


def image2b64(image_file):
    with open(image_file, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    return image_b64


def text2audio(text):
    edge_command = f'edge-tts --text "{text}" --write-media ./content/audio.mp3'
    os.system(edge_command)
    Audio('./content/audio.mp3', autoplay=True)
    return None


def extract_python_code(text):
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def execute_and_return(x):
    code = extract_python_code(x.content)[0]
    try:
        result = exec(str(code))
    except ExceptionType:
        print("The code is not executable, don't give up, try again!")
    return x


def execute_and_return_gr(x):
    code = extract_python_code(x.content)[0]
    try:
        result = exec(str(code))
    except ExceptionType:
        print("The code is not executable, don't give up, try again!")
    return img_path


def model_process(image_b64, user_input, table):
    image_b64 = image2b64(image_b64)
    chart_reading = ChatNVIDIA(model="microsoft/phi-3-vision-128k-instruct")
    chart_reading_prompt = ChatPromptTemplate.from_template(
        'Generate underlying data table of the figure below, : <img src="data:image/png;base64,{image_b64}" />'
    )
    chart_chain = chart_reading_prompt | chart_reading
    instruct_chat = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
    instruct_prompt = ChatPromptTemplate.from_template(
        "Do NOT repeat my requirements already stated. Based on this table {table}, {input}" \
        "If has table string, start with 'TABLE', end with 'END_TABLE'." \
        "If has code, start with '```python' and end with '```'." \
        "Do NOT include table inside code, and vice versa."
    )
    instruct_chain = instruct_prompt | instruct_chat
    chart_reading_branch = RunnableBranch(
        (lambda x: x.get('table') is None, RunnableAssign({'table': chart_chain})),
        (lambda x: x.get('table') is not None, lambda x: x),
        lambda x: x
    )
    update_table = RunnableBranch(
        (lambda x: 'TABLE' in x.content, save_table_to_global),
        lambda x: x
    )
    execute_code = RunnableBranch(
        (lambda x: '```python' in x.content, execute_and_return_gr),
        lambda x: x
    )
    chain = (
            chart_reading_branch
            | instruct_chain
            | update_table
            | execute_code
    )
    return chain.invoke({"image_b64": image_b64, "input": user_input, "table": table})


def main():
    get_nvidia_key()
    model = "meta/llama-3.1-405b-instruct"
    embedder = "NV-Embed-QA"
    # model_initial(model, embedder)
    gradio_interface = gradio.Interface(fn=model_process,
                                        inputs=[gradio.Image(label="Upload image", type="filepath"), 'text'],
                                        outputs=['image'],
                                        title="Multi Modal chat agent",
                                        description="Multi Modal chat agent",
                                        allow_flagging="never")
    gradio_interface.launch(debug=True, share=False, show_api=False)
    return None


if __name__ == "__main__":
    main()
```
