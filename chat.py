import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os

# 模型参数设置
# 设置 GPU 编号
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
MODEL_PATH = "THUDM/glm-4v-9b"

# 检查是否有可用的 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 加载模型，并指定 device_map 参数
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
).eval()

# 生成文本的参数
gen_kwargs = {
    "max_length": 2500,
    "do_sample": True,
    "top_k": 1,
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "generation_config": GenerationConfig(progress_bar=True)
}


def chat(query: str, history=None, image_path=None):
    # 对话循环
    while True:
        # 如果用户指令中存在image_path,说明是带图的
        if image_path:  # 如果用户指令中存在,说明是带图的
            # 将用户输入添加到对话历史
            image = Image.open(image_path).convert('RGB')
            history.append({"role": "user", "image_path": image, "content": query})
        else:  # 否则是纯对话
            # 将用户输入添加到对话历史
            history.append({"role": "user", "content": query})

        # 使用对话历史构建输入
        inputs = tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                               return_dict=True)

        # 生成响应
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            response_ids = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # 将模型生成的响应添加到对话历史
        history.append({"role": "assistant", "content": response})
        return response
