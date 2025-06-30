import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont



# 添加角点约束损失函数
def rectangle_constraint_loss(corners):
    """计算四个角点构成矩形的约束损失（相邻边应该垂直）"""
    # 重塑为 [batch_size, 4, 2]
    corners = corners.view(-1, 4, 2)
    batch_size = corners.size(0)
    
    # 计算四条边的向量
    edges = []
    for i in range(4):
        next_i = (i + 1) % 4
        edge = corners[:, next_i, :] - corners[:, i, :]
        edges.append(edge)
    
    # 计算相邻边的点积，对于矩形，相邻边应垂直，点积应为0
    loss = 0
    for i in range(4):
        next_i = (i + 1) % 4
        dot_product = torch.sum(edges[i] * edges[next_i], dim=1)
        loss += torch.mean(dot_product ** 2)
    
    return loss / 4

def trapezoid_constraint_loss(corners):
    """计算四个角点构成梯形的约束损失（对边应该平行）"""
    # 重塑为 [batch_size, 4, 2]
    corners = corners.view(-1, 4, 2)
    batch_size = corners.size(0)
    
    # 计算四条边的向量
    edges = []
    for i in range(4):
        next_i = (i + 1) % 4
        edge = corners[:, next_i, :] - corners[:, i, :]
        edges.append(edge)
    
    # 对边应平行，计算叉积，平行时叉积为0
    cross_product1 = edges[0][:, 0] * edges[2][:, 1] - edges[0][:, 1] * edges[2][:, 0]
    cross_product2 = edges[1][:, 0] * edges[3][:, 1] - edges[1][:, 1] * edges[3][:, 0]
    
    # 平行度损失
    loss = torch.mean(cross_product1 ** 2) + torch.mean(cross_product2 ** 2)
    
    return loss / 2

def parallelogram_constraint_loss(corners):
    """计算四个角点构成平行四边形的约束损失（对边应该平行且相等）"""
    # 重塑为 [batch_size, 4, 2]
    corners = corners.view(-1, 4, 2)
    batch_size = corners.size(0)
    
    # 计算四条边的向量
    edges = []
    for i in range(4):
        next_i = (i + 1) % 4
        edge = corners[:, next_i, :] - corners[:, i, :]
        edges.append(edge)
    
    # 对边应平行，计算叉积
    cross_product1 = edges[0][:, 0] * edges[2][:, 1] - edges[0][:, 1] * edges[2][:, 0]
    cross_product2 = edges[1][:, 0] * edges[3][:, 1] - edges[1][:, 1] * edges[3][:, 0]
    
    # 对边应相等，计算长度差
    len_diff1 = torch.sum((torch.norm(edges[0], dim=1) - torch.norm(edges[2], dim=1)) ** 2)
    len_diff2 = torch.sum((torch.norm(edges[1], dim=1) - torch.norm(edges[3], dim=1)) ** 2)
    
    # 平行度损失 + 长度差异损失
    loss = torch.mean(cross_product1 ** 2) + torch.mean(cross_product2 ** 2) + len_diff1 + len_diff2
    
    return loss / 4

# 梯形注意力模块
class TrapezoidAttention(nn.Module):
    def __init__(self, in_channels):
        super(TrapezoidAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attention = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # 方向预测（上宽或下宽）
        self.direction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 基础特征处理
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        
        # 预测梯形方向（0表示上宽，1表示下宽）
        direction = self.direction(x)
        
        # 根据方向生成注意力掩码
        b, c, h, w = feat.size()
        attention = self.attention(feat)
        
        # 创建梯形注意力掩码
        mask = torch.ones_like(attention)
        for i in range(b):
            # 如果direction > 0.5，则梯形下部宽（仰视），增强下部
            # 否则梯形上部宽（俯视），增强上部
            if direction[i, 0, 0, 0] > 0.5:
                # 仰视角 - 增强下部
                for j in range(h):
                    weight = 0.5 + 0.5 * (j / h)  # 从上到下权重增加
                    mask[i, :, j, :] *= weight
            else:
                # 俯视角 - 增强上部
                for j in range(h):
                    weight = 1.0 - 0.5 * (j / h)  # 从上到下权重减少
                    mask[i, :, j, :] *= weight
        
        # 应用注意力掩码
        attention = attention * mask
        attention = self.sigmoid(attention)
        
        # 返回增强后的特征和方向
        return x * attention, direction

# 平行四边形注意力模块
class ParallelogramAttention(nn.Module):
    def __init__(self, in_channels):
        super(ParallelogramAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 使用可变形卷积来适应倾斜结构
        try:
            from torchvision.ops import deform_conv2d
            self.use_deform = True
            self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
            self.deform_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        except:
            print("警告: 无法导入可变形卷积，将使用标准卷积替代")
            self.use_deform = False
            self.deform_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        self.attention = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 基础特征处理
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        
        # 使用可变形卷积处理特征（如果可用）
        if self.use_deform:
            offset = self.offset_conv(feat)
            from torchvision.ops import deform_conv2d
            deform_feat = deform_conv2d(feat, offset, self.deform_conv.weight, self.deform_conv.bias, padding=1)
        else:
            deform_feat = self.deform_conv(feat)
        
        # 生成注意力图
        attention = self.attention(deform_feat)
        attention = self.sigmoid(attention)
        
        # 返回增强后的特征
        return x * attention

class PerspectiveCorrectionNetwork(nn.Module):
    """
    透视矫正网络 - 对输入图像进行透视变换矫正，改进为三分支版本
    三个分支: 矩形分支、梯形分支、平行四边形分支
    支持使用VGG16或ResNet50作为特征提取器
    """
    def __init__(self, backbone='vgg16', in_channels=3, num_distortion_classes=3):
        super(PerspectiveCorrectionNetwork, self).__init__()
        
        # 设置固定的分类数为3
        num_distortion_classes = 3
        
        # 特征提取器选择
        if backbone == 'vgg16':
            # 使用预训练的VGG16模型作为编码器
            base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            
            # 修改第一层以接受任意通道数的输入
            if in_channels != 3:
                base_model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
            
            # 只使用VGG16的特征提取部分
            self.encoder = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d(1)
            )
            
            # VGG16最后一层特征通道数为512
            feature_dim = 512
            
        elif backbone == 'resnet50':
            # 使用预训练的ResNet50模型作为编码器
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            
            # 修改第一层以接受任意通道数的输入
            if in_channels != 3:
                base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # 移除分类层，只保留特征提取部分
            self.encoder = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
                nn.AdaptiveAvgPool2d(1)
            )
            
            # ResNet50最后一层特征通道数为2048
            feature_dim = 2048
            
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}，请选择 'vgg16' 或 'resnet50'")
        
        # 添加一个分类分支用于判断透视变换类型
        # 类别包括：矩形、梯形、平行四边形
        self.distortion_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_distortion_classes)
        )
        
        # 三个专门的角点预测分支
        # 1. 矩形分支 - 用于正常视角
        self.rectangle_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 8)  # 输出 [x1, y1, x2, y2, x3, y3, x4, y4] 坐标
        )
        
        # 2. 梯形分支 - 用于俯视角和仰视角
        self.trapezoid_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 9)  # 8个坐标 + 1个方向参数（0表示上宽，1表示下宽）
        )
        
        # 3. 平行四边形分支 - 用于倾斜视角
        self.parallelogram_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 8)  # 输出 [x1, y1, x2, y2, x3, y3, x4, y4] 坐标
        )
        
        # 初始化分类器权重
        self.distortion_classifier[-1].weight.data.normal_(0, 0.01)
        
        # 初始化矩形预测器权重
        self.rectangle_predictor[-1].weight.data.normal_(0, 0.01)
        bias_tensor = torch.tensor([
            0.2, 0.2,   # 左上角
            0.8, 0.2,   # 右上角
            0.8, 0.8,   # 右下角
            0.2, 0.8,   # 左下角
        ], dtype=torch.float)
        self.rectangle_predictor[-1].bias.data.copy_(bias_tensor)
        
        # 初始化梯形预测器权重
        self.trapezoid_predictor[-1].weight.data.normal_(0, 0.01)
        # 梯形偏置包含方向信息
        bias_tensor = torch.tensor([
            0.15, 0.2,   # 左上角
            0.85, 0.2,   # 右上角
            0.75, 0.8,   # 右下角
            0.25, 0.8,   # 左下角
            0.5          # 方向参数（0.5表示中性）
        ], dtype=torch.float)
        self.trapezoid_predictor[-1].bias.data.copy_(bias_tensor)
        
        # 初始化平行四边形预测器权重
        self.parallelogram_predictor[-1].weight.data.normal_(0, 0.01)
        bias_tensor = torch.tensor([
            0.2 + 0.05, 0.2,     # 左上角（轻微右移）
            0.8, 0.2 + 0.05,     # 右上角（轻微下移）
            0.8 - 0.05, 0.8,     # 右下角（轻微左移）
            0.2, 0.8 - 0.05,     # 左下角（轻微上移）
        ], dtype=torch.float)
        self.parallelogram_predictor[-1].bias.data.copy_(bias_tensor)
        
        # 自适应融合不同角点预测结果的权重层
        self.fusion_weights = nn.Sequential(
            nn.Linear(num_distortion_classes, num_distortion_classes),
            nn.Softmax(dim=1)
        )
        
        # 存储所使用的骨干网络名称和分类器类别数量
        self.backbone = backbone
        self.num_distortion_classes = num_distortion_classes
        
        # 添加特殊的注意力模块
        self.trapezoid_attention = TrapezoidAttention(256)
        self.parallelogram_attention = ParallelogramAttention(256)

    def forward(self, x, target_corners=None, keep_aspect_ratio=True):
        batch_size, channels, height, width = x.size()
        
        # 提取特征
        features = self.encoder(x)
        
        # 分类透视变换类型（矩形、梯形、平行四边形）
        distortion_logits = self.distortion_classifier(features)
        distortion_probs = torch.softmax(distortion_logits, dim=1)
        
        # 预测各类型的角点坐标
        # 1. 矩形预测
        rectangle_output = torch.sigmoid(self.rectangle_predictor(features))
        rectangle_corners = rectangle_output
        
        # 2. 梯形预测
        trapezoid_output = torch.sigmoid(self.trapezoid_predictor(features))
        trapezoid_corners = trapezoid_output[:, :8]  # 前8个元素是角点坐标
        trapezoid_direction = trapezoid_output[:, 8:9]  # 最后一个元素是方向参数
        
        # 3. 平行四边形预测
        parallelogram_corners = torch.sigmoid(self.parallelogram_predictor(features))
        
        # 根据分类概率融合不同类型的角点预测
        fusion_weights = self.fusion_weights(distortion_probs)  # [batch_size, num_classes]
        
        # 初始化融合后的角点预测
        pred_corners = torch.zeros_like(rectangle_corners)
        
        # 加权融合各类别预测的角点
        all_corners = [rectangle_corners, trapezoid_corners, parallelogram_corners]
        for i in range(self.num_distortion_classes):
            weight = fusion_weights[:, i].view(-1, 1)  # [batch_size, 1]
            pred_corners += weight * all_corners[i]
        
        # 训练时完全使用真实标注生成透视变换图像
        # 但模型仍然通过比较预测和真实标注计算损失
        if self.training and target_corners is not None:
            # 100%使用真实标注进行透视变换
            use_corners = target_corners
        else:
            # 测试/推理时使用预测角点
            use_corners = pred_corners
        
        # 创建用于存储变换后图像的张量
        transformed_x = torch.zeros_like(x)
        
        # 转换相对坐标到像素坐标，并应用透视变换
        for i in range(batch_size):
            # 将角点重塑为[4,2]的形状
            corner_coords = use_corners[i].view(4, 2)
            
            # 转换为像素坐标
            src_points = corner_coords * torch.tensor([width, height], device=use_corners.device)
            src_points = src_points.detach().cpu().numpy().astype(np.float32)
            
            # 计算保持长宽比的目标矩形
            if keep_aspect_ratio:
                # 计算源区域的边长
                top_edge = np.linalg.norm(src_points[1] - src_points[0])
                right_edge = np.linalg.norm(src_points[2] - src_points[1])
                bottom_edge = np.linalg.norm(src_points[3] - src_points[2])
                left_edge = np.linalg.norm(src_points[0] - src_points[3])
                
                # 计算平均宽度和高度
                avg_width = (top_edge + bottom_edge) / 2
                avg_height = (left_edge + right_edge) / 2
                
                # 计算目标矩形的尺寸，保持长宽比
                target_width = min(width, int(avg_width))
                target_height = min(height, int(avg_height))
                
                # 计算偏移量，使矩形居中
                offset_x = (width - target_width) // 2
                offset_y = (height - target_height) // 2
                
                # 设置目标矩形的四个角点
                dst_points = np.array([
                    [offset_x, offset_y],                      # 左上
                    [offset_x + target_width, offset_y],       # 右上
                    [offset_x + target_width, offset_y + target_height], # 右下
                    [offset_x, offset_y + target_height]       # 左下
                ], dtype=np.float32)
            else:
                # 不保持长宽比时，直接使用图像四个角点作为目标
                dst_points = np.array([
                    [0, 0],              # 左上
                    [width, 0],          # 右上
                    [width, height],     # 右下
                    [0, height]          # 左下
                ], dtype=np.float32)
            
            try:
                # 计算透视变换矩阵
                M = cv2.getPerspectiveTransform(src_points, dst_points)
                
                # 将图像转换为numpy格式
                img_np = x[i].permute(1, 2, 0).cpu().numpy()
                
                # 确保图像格式正确
                if img_np.dtype != np.uint8 and img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                
                # 应用透视变换
                warped = cv2.warpPerspective(
                    img_np, M, (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                
                # 转换回PyTorch张量
                if warped.dtype == np.uint8:
                    warped = warped.astype(np.float32) / 255.0
                transformed_x[i] = torch.from_numpy(warped).permute(2, 0, 1).to(x.device)
            except Exception as e:
                print(f"透视变换错误: {e}")
                # 发生错误时，使用原始图像
                transformed_x[i] = x[i]
        
        # 返回变换后的图像、预测的角点、分类结果以及附加信息
        return transformed_x, pred_corners, distortion_probs, {
            'trapezoid_direction': trapezoid_direction,
            'rectangle_corners': rectangle_corners,
            'trapezoid_corners': trapezoid_corners,
            'parallelogram_corners': parallelogram_corners
        }

    def compute_geometry_losses(self, pred_corners):
        """计算几何约束损失"""
        # 从预测角点的额外信息中获取各分支角点
        rectangle_corners = pred_corners['rectangle_corners'] if isinstance(pred_corners, dict) else pred_corners
        trapezoid_corners = pred_corners['trapezoid_corners'] if isinstance(pred_corners, dict) else pred_corners
        parallelogram_corners = pred_corners['parallelogram_corners'] if isinstance(pred_corners, dict) else pred_corners
        
        # 应用几何约束损失到各个分支的角点预测
        rectangle_loss = rectangle_constraint_loss(rectangle_corners)
        trapezoid_loss = trapezoid_constraint_loss(trapezoid_corners)
        parallelogram_loss = parallelogram_constraint_loss(parallelogram_corners)
        
        return {
            'rectangle_constraint_loss': rectangle_loss,
            'trapezoid_constraint_loss': trapezoid_loss,
            'parallelogram_constraint_loss': parallelogram_loss
        }

# 辅助函数：确保四个角点按照左上、右上、右下、左下的顺序排列
def reorder_quad_points(points):
    """
    根据坐标值重新排序四边形的四个点，使其符合左上、右上、右下、左下的顺序
    
    Args:
        points: numpy数组，形状为[4, 2]，表示四个角点的(x,y)坐标
        
    Returns:
        排序后的点，numpy数组，形状为[4, 2]
    """
    # 计算中心点
    center = np.mean(points, axis=0)
    
    # 根据点到中心点的角度排序
    def compute_angle(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0])
    
    # 计算每个点的角度
    angles = [compute_angle(point) for point in points]
    
    # 找出左上角点（角度最接近-π或π）
    top_left_idx = np.argmin([angle if angle >= 0 else angle + 2*np.pi for angle in angles])
    
    # 按照顺时针顺序排列点
    ordered_points = []
    for i in range(4):
        idx = (top_left_idx + i) % 4
        ordered_points.append(points[idx])
    
    return np.array(ordered_points)

# 辅助函数：在图像上绘制预测的四边形区域
def draw_region_box(image_tensor, corners, message=None, debug_info=False, distortion_probs=None):
    """
    在图像上绘制预测的四边形区域
    
    Args:
        image_tensor (torch.Tensor): 形状为 [C, H, W] 的图像张量
        corners (torch.Tensor): 形状为 [8] 的张量，包含 [x1, y1, x2, y2, x3, y3, x4, y4] 坐标 (相对值0-1)
                               顺序为：左上、右上、右下、左下
        message (str, optional): 要在图像上显示的文本信息
        debug_info (bool, optional): 是否显示调试信息
        distortion_probs (torch.Tensor, optional): 形状为 [num_classes] 的张量，包含透视变换类型的概率
    
    Returns:
        torch.Tensor: 带有四边形标记的图像张量，形状为 [C, H, W]
    """
    # 转换为numpy数组并转置为HWC格式
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # 确保图像值在正确范围
    if image.max() <= 1.0:
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
    
    # 转换为OpenCV格式进行绘制
    if image.shape[2] == 1:  # 灰度图像
        cv_image = np.repeat(image, 3, axis=2)
    else:  # RGB图像
        cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 获取图像尺寸
    height, width = cv_image.shape[:2]
    
    # 将角点坐标转换为像素值
    corner_points = corners.cpu().numpy().reshape(4, 2)
    corner_points[:, 0] *= width
    corner_points[:, 1] *= height
    corner_points = corner_points.astype(np.int32)
    
    # 绘制四边形区域
    cv2.polylines(cv_image, [corner_points], True, (0, 255, 0), 2)
    
    # 绘制四个角点
    for i, (x, y) in enumerate(corner_points):
        cv2.circle(cv_image, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(cv_image, f"{i+1}", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # 添加文本信息
    if message is not None:
        # 使用支持中文的PIL绘制文本
        pil_img = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 尝试使用不同的中文字体
        try:
            # Windows系统上常见的中文字体
            fontpath = "simhei.ttf"  # 黑体
            if not os.path.exists(fontpath):
                fontpath = "simsun.ttc"  # 宋体
            if not os.path.exists(fontpath):
                fontpath = "msyh.ttc"  # 微软雅黑
            if not os.path.exists(fontpath):
                # 如果上述字体都不存在，尝试使用系统字体
                import matplotlib.font_manager as fm
                fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
                # 查找可能的中文字体
                for font in fonts:
                    if 'simhei' in font.lower() or 'simsun' in font.lower() or 'msyh' in font.lower():
                        fontpath = font
                        break
                else:
                    # 如果找不到中文字体，使用默认字体
                    fontpath = fm.findfont(fm.FontProperties())
            
            font = ImageFont.truetype(fontpath, 24)
            # 在图像上绘制文本
            draw.text((10, 10), message, font=font, fill=(255, 0, 0))
        except Exception as e:
            print(f"绘制中文文本失败: {e}")
            # 如果失败，回退到OpenCV绘制
            cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2.putText(cv_image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # 如果成功，使用PIL绘制的图像
            cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # 如果有分类概率信息，显示在图像上
    if distortion_probs is not None:
        # 英文标签，避免中文问题
        distortion_types = ["Normal View", "Top-Down View", "Bottom-Up View", "Tilted View"]
        probs = distortion_probs.cpu().numpy()
        
        # 找出概率最高的类别
        max_prob_idx = np.argmax(probs)
        max_prob_value = probs[max_prob_idx]
        
        # 显示分类信息
        class_info = f"{distortion_types[max_prob_idx]} ({max_prob_value:.2f})"
        cv2.putText(cv_image, class_info, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示详细概率信息
        if debug_info:
            for i, (dtype, prob) in enumerate(zip(distortion_types, probs)):
                text = f"{dtype}: {prob:.2f}"
                cv2.putText(cv_image, text, (width - 200, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    # 转换回PyTorch张量
    result = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    result = torch.from_numpy(result.transpose(2, 0, 1).astype(np.float32) / 255.0)
    
    return result

# 辅助函数：创建标准透视变换图像
def create_standard_transform(image_tensor, corners, target_size=None, fixed_target_points=None):
    """
    根据给定的角点创建标准透视变换图像，该图像是仅根据真实标注生成的，不使用模型预测结果
    
    Args:
        image_tensor (torch.Tensor): 形状为 [C, H, W] 的图像张量
        corners (torch.Tensor): 形状为 [8] 的张量，包含 [x1, y1, x2, y2, x3, y3, x4, y4] 坐标 (相对值0-1)
                              顺序为：左上、右上、右下、左下
        target_size (tuple, optional): 目标尺寸(宽,高)，如果为None则使用原图尺寸
        fixed_target_points (np.ndarray, optional): 预定义的目标点坐标，形状为[4, 2]
                              
    Returns:
        torch.Tensor: 标准透视变换后的图像张量，形状为 [C, H, W]
    """
    # 确保图像是3D张量 [C, H, W]
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    
    # 获取图像尺寸
    channels, height, width = image_tensor.shape
    
    # 将角点重塑为[4,2]形状
    corner_coords = corners.view(4, 2)
    
    # 转换为像素坐标
    src_points = corner_coords * torch.tensor([width, height], device=corners.device)
    src_points = src_points.detach().cpu().numpy().astype(np.float32)
    
    # 设置目标尺寸
    if target_size:
        output_width, output_height = target_size
    else:
        output_width, output_height = width, height
    
    # 设置目标矩形的四个角点
    if fixed_target_points is not None:
        dst_points = fixed_target_points
    else:
        # 计算源区域的边长
        top_edge = np.linalg.norm(src_points[1] - src_points[0])
        right_edge = np.linalg.norm(src_points[2] - src_points[1])
        bottom_edge = np.linalg.norm(src_points[3] - src_points[2])
        left_edge = np.linalg.norm(src_points[0] - src_points[3])
        
        # 计算平均宽度和高度
        avg_width = (top_edge + bottom_edge) / 2
        avg_height = (left_edge + right_edge) / 2
        
        # 使用保持长宽比的目标矩形
        target_width = min(output_width, int(avg_width))
        target_height = min(output_height, int(avg_height))
        
        # 计算偏移量，使矩形居中
        offset_x = (output_width - target_width) // 2
        offset_y = (output_height - target_height) // 2
        
        # 设置目标矩形
        dst_points = np.array([
            [offset_x, offset_y],                      # 左上
            [offset_x + target_width, offset_y],       # 右上
            [offset_x + target_width, offset_y + target_height], # 右下
            [offset_x, offset_y + target_height]       # 左下
        ], dtype=np.float32)
    
    try:
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 将图像转换为numpy格式
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # 确保图像格式正确
        if img_np.dtype != np.uint8 and img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        # 应用透视变换
        warped = cv2.warpPerspective(
            img_np, M, (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # 转换回PyTorch张量
        if warped.dtype == np.uint8:
            warped = warped.astype(np.float32) / 255.0
        return torch.from_numpy(warped).permute(2, 0, 1).to(image_tensor.device)
    except Exception as e:
        print(f"标准透视变换错误: {e}")
        # 发生错误时，使用原始图像
        return image_tensor 