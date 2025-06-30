import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from perspective_corrector import PerspectiveCorrectionNetwork, draw_region_box
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='批量测试透视矫正网络')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='特征提取器骨干网络')
    parser.add_argument('--output_dir', type=str, default='test_results', help='输出目录')
    parser.add_argument('--corners', type=float, nargs=8, default=[0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8], 
                        help='目标角点坐标 [左上x,左上y, 右上x,右上y, 右下x,右下y, 左下x,左下y]')
    parser.add_argument('--extensions', type=str, nargs='+', default=['.jpg', '.jpeg', '.png'], 
                        help='支持的图像扩展名')
    return parser.parse_args()

def process_image(model, image_path, corners, transform, device, output_dir):
    # 加载图像
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"无法加载图像 {image_path}: {e}")
        return
    
    # 准备输入
    image_tensor = transform(image).unsqueeze(0).to(device)
    corners_tensor = torch.tensor(corners, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 进行推理
    with torch.no_grad():
        transformed_image, predicted_corners, distortion_probs, _ = model(image_tensor, corners_tensor)
    
    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    orig_img = image_tensor[0] * std + mean
    orig_img = torch.clamp(orig_img, 0, 1)
    transformed_image = transformed_image[0] * std + mean
    transformed_image = torch.clamp(transformed_image, 0, 1)
    
    # 获取分类结果
    distortion_types = ["Normal View", "Top-Down View", "Bottom-Up View", "Tilted View"]
    max_prob_idx = torch.argmax(distortion_probs[0]).item()
    predicted_type = distortion_types[max_prob_idx]
    
    # 画检测框
    pred_corners = predicted_corners[0].cpu()
    orig_with_box = draw_region_box(orig_img.cpu(), pred_corners, distortion_probs=distortion_probs[0])
    
    # 转为PIL
    orig_pil = transforms.ToPILImage()(orig_img.cpu())
    orig_with_box_pil = transforms.ToPILImage()(orig_with_box.cpu())
    result_image = transforms.ToPILImage()(transformed_image.cpu())
    
    # 创建三联图
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(orig_pil)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(orig_with_box_pil)
    plt.title(f'Detection Box: {predicted_type}')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result_image)
    plt.title('Transformed Image')
    plt.axis('off')
    
    plt.tight_layout()
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f'result_{base_name}')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存变换后的图像
    result_image.save(os.path.join(output_dir, f'transformed_{base_name}'))
    
    return predicted_corners[0].cpu().numpy()

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = PerspectiveCorrectionNetwork(backbone=args.backbone).to(device)
    print(f"创建透视矫正网络，使用 {args.backbone} 作为特征提取器")
    
    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 如果加载的是检查点文件，提取模型状态字典
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 如果直接加载的是模型权重
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"已加载模型权重: {args.model_path}")
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 获取所有图像文件
    image_files = []
    for ext in args.extensions:
        image_files.extend([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                          if f.lower().endswith(ext)])
    
    if not image_files:
        print(f"在 {args.input_dir} 中没有找到支持的图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理所有图像
    all_corners = []
    for image_path in tqdm(image_files, desc="处理图像"):
        corners = process_image(model, image_path, args.corners, transform, device, args.output_dir)
        if corners is not None:
            all_corners.append(corners)
    
    # 计算平均角点坐标
    if all_corners:
        avg_corners = np.mean(all_corners, axis=0)
        print("\n平均预测角点坐标:")
        print(f"左上: ({avg_corners[0]:.4f}, {avg_corners[1]:.4f})")
        print(f"右上: ({avg_corners[2]:.4f}, {avg_corners[3]:.4f})")
        print(f"右下: ({avg_corners[4]:.4f}, {avg_corners[5]:.4f})")
        print(f"左下: ({avg_corners[6]:.4f}, {avg_corners[7]:.4f})")
    
    print(f"\n所有结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 