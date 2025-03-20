import torch
from vgg import vgg

def load_model(model_path):
    checkpoint = torch.load(model_path)
    # 根据剪枝后的模型重新配置卷积参数，剪枝后的参数配置保存在checkpoint['cfg']
    model = vgg(cfg=checkpoint['cfg'])
    return model

model = load_model("pruned.pth.tar")
dummy_input = torch.randn(1, 3, 32, 32)  

torch.onnx.export(model,
                  dummy_input,
                  "prune_model.onnx", 
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

print("ONNX模型已导出！")
