import torch
import time
from torchsummary import summary
from perspective_corrector import PerspectiveCorrectionNetwork
from ptflops import get_model_complexity_info

# 初始化模型
model = PerspectiveCorrectionNetwork(backbone='vgg16')
model.cuda()
model.eval()  # 测试模式

# 打印参数量
summary(model, input_size=(3, 224, 224))

# 计算 FLOPs 和 参数量
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print(f"\n[计算复杂度]")
    print(f'FLOPs: {macs}')  # MACs 是 Multiply–Accumulate operations ≈ FLOPs
    print(f'Params: {params}')

# 计算前向传播时间（Forward Time）
dummy_input = torch.randn(1, 3, 224, 224).cuda()
torch.cuda.synchronize()
start_time = time.time()
for _ in range(100):  # 多次前向传播平均值更稳定
    _ = model(dummy_input)
torch.cuda.synchronize()
end_time = time.time()

avg_time = (end_time - start_time) / 100
fps = 1 / avg_time

print(f"\n[推理时间]")
print(f'Average Forward Time: {avg_time:.5f} s')
print(f'FPS: {fps:.2f}')
