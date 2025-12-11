import torch
import torch.nn as nn
from thop import profile, clever_format

from backbone.pvtv2 import pvt_v2_b2
# from transformer_decoder import transfmrerDecoder
import torch.nn.functional as F
# from model.MultiScaleAttention import Block

# from fvcore.nn import FlopCountAnalysis, parameter_count_table
import numpy as np 
import cv2






class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.relu(x)    # added by lzg on 2025.6.8
        return x




class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x




class decoder1(nn.Module):
    def __init__(self, channels):
        super(decoder1, self).__init__()
        self.convf = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(2*channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(3*channels, channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(4*channels, channels, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
    def forward(self, x3, x4, x5, x6):
        x3 = self.upsample(x3)
        x4 = self.conv4(torch.cat((x3, x4), dim=1))
        x3 = self.upsample(x3)
        x4 = self.upsample(x4)
        x5 = self.conv5(torch.cat((x3, x4, x5), dim=1))
        x5 = self.upsample(x5)
        x4 = self.upsample(x4)
        x3 = self.upsample(x3)
        x6 = self.conv6(torch.cat((x3, x4, x5, x6), dim=1))

        return x6


class decoder2(nn.Module):
    def __init__(self, channels):
        super(decoder2, self).__init__()
        self.convf = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(2*channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(3*channels, channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(4*channels, channels, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
    def forward(self, x3, x4, x5, x6):
        x3 = self.upsample(x3)
        x4 = self.conv4(torch.cat((x3, x4), dim=1))
        x3 = self.upsample(x3)
        x4 = self.upsample(x4)
        x5 = self.conv5(torch.cat((x3, x4, x5), dim=1))
        x5 = self.upsample(x5)
        x4 = self.upsample(x4)
        x3 = self.upsample(x3)
        x6 = self.conv6(torch.cat((x3, x4, x5, x6), dim=1))

        return x6

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class Rgb_guide_sa(nn.Module):
    def __init__(self):
        super(Rgb_guide_sa, self).__init__()
        self.SA = SpatialAttention()


    def forward(self, rgb, focal, focal2):
        rgb_sa = self.SA(rgb)
        focal_sa = torch.mul(rgb_sa, focal)
        focal2 = F.interpolate(focal2, scale_factor=2, mode='bilinear', align_corners=False)
        focal = focal_sa + focal +focal2

        return focal

# --- 新的打包模块 ---
class SimpleFusionModule(nn.Module):
    def __init__(self, num_streams, channels_per_stream, out_channels_stage1, out_channels_stage2):
        super(SimpleFusionModule, self).__init__()
        fusion_input_channels = num_streams * channels_per_stream
        self.fusion_conv1 = BasicConv2d(fusion_input_channels, out_channels_stage1, kernel_size=3, padding=1)
        self.fusion_conv2 = BasicConv2d(out_channels_stage1, out_channels_stage2, kernel_size=3, padding=1)

    def forward(self, list_of_features):
        concatenated_features = torch.cat(list_of_features, dim=1)
        fused_output = self.fusion_conv1(concatenated_features)
        fused_output = self.fusion_conv2(fused_output)
        return fused_output

# class SimpleDecoderModule(nn.Module):
#     def __init__(self, in_channels, num_classes=1, upsample_factor=32):
#         super(SimpleDecoderModule, self).__init__()
#         self.final_pred_conv = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
#         self.upsampler = nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False)

#     def forward(self, fused_features):
#         saliency_logits = self.final_pred_conv(fused_features)
#         saliency_map_logits = self.upsampler(saliency_logits)
#         return saliency_map_logits

class Customdfrm(nn.Module):
    def __init__(self, in_channels_of, in_channels_rgb, out_channels_stage1, out_channels_stage2):
        super(Customdfrm, self).__init__()
        # 假设 OF1, OF2 具有 'in_channels_of' 通道数
        # RGB 具有 'in_channels_rgb' 通道数
        # 在你的情况下，这些都将是 'rfb_out_channels' (例如, 32)
        fusion_input_channels = 2 * in_channels_of + in_channels_rgb
        self.fusion_conv1 = BasicConv2d(fusion_input_channels, out_channels_stage1, kernel_size=3, padding=1)
        self.fusion_conv2 = BasicConv2d(out_channels_stage1, out_channels_stage2, kernel_size=3, padding=1)

    def forward(self, of_feat1, of_feat2, rgb_feat):
        # 沿通道维度 (dim=1) 拼接
        concatenated_features = torch.cat([of_feat1, of_feat2, rgb_feat], dim=1)
        fused_output = self.fusion_conv1(concatenated_features)
        fused_output = self.fusion_conv2(fused_output)
        return fused_output

class DFRM(nn.Module):
    def __init__(self, channels):
        super(DFRM, self).__init__()
        self.channels = channels

        # 左侧：质量分数 Q1, Q2 生成模块 (MLP + Sigmoid)
        # MLP 可以用两个1x1卷积实现：Conv -> ReLU -> Conv
        # 输入是两个特征图拼接，所以是 2*channels
        # 输出是1个通道的质量图 (B, 1, H, W)
        mlp_hidden_channels = channels // 2  # 可调参数
        self.quality_mlp = nn.Sequential(
            nn.Conv2d(2 * channels, mlp_hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_hidden_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 右侧：Fr 和 For_ 交互 (通道注意力)
        self.gap = nn.AdaptiveAvgPool2d(1) # Global Average Pooling

        # FC 层用 1x1 卷积实现，方便处理 (B, C, 1, 1) 的通道注意力权重
        # 包含一个压缩和激励结构 (Squeeze-and-Excitation)
        fc_squeeze_channels = channels // 4 # 可调参数
        self.fc_fr_for_channel_attention = nn.Sequential(
            nn.Conv2d(channels, fc_squeeze_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(fc_squeeze_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.fc_for_channel_attention = nn.Sequential(
            nn.Conv2d(channels, fc_squeeze_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(fc_squeeze_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 右侧：Fr 和 Fo 交互 (空间注意力)
        # AP 和 MP 之后拼接，输入到 Conv7 的是 2 个通道
        # Conv7 输出 1 个通道的空间注意力图
        # 这里的 AP 和 MP 指的是在通道维度上进行池化，得到 (B, 1, H, W)
        self.spatial_attention_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_attention_sigmoid = nn.Sigmoid()


    def forward(self, Fo, Fr, For_): # Fo: 光流1, Fr: RGB, For_: 光流2

        # --- 左侧逻辑：生成质量分数 Q1 和 Q2 ---
        # Q1 (Fr 和 Fo 交互)
        q1_input = torch.cat([Fr, Fo], dim=1)
        Q1 = self.quality_mlp(q1_input)  # Shape: (B, 1, H, W)

        # Q2 (Fr 和 For_ 交互)
        q2_input = torch.cat([Fr, For_], dim=1)
        Q2 = self.quality_mlp(q2_input)  # Shape: (B, 1, H, W)


        # --- 右侧逻辑：Fr 和 For_ (底部光流) 交互 ---
        # 1. 计算通道注意力
        ca_fr_for_for = self.fc_fr_for_channel_attention(self.gap(Fr))    # Shape: (B, C, 1, 1)
        ca_for = self.fc_for_channel_attention(self.gap(For_))          # Shape: (B, C, 1, 1)

        # 2. 平均通道注意力
        avg_ca_for_for = (ca_fr_for_for + ca_for) / 2.0

        # 3. 通道注意力应用于 For_
        for_channel_attended = For_ * avg_ca_for_for # 广播机制: (B,C,H,W) * (B,C,1,1)

        # 4. 应用质量分数 Q2
        #    Fr 和 For_ 交互的最终特征
        final_feat_fr_for = for_channel_attended * Q2 # 广播机制: (B,C,H,W) * (B,1,H,W)


        # --- 右侧逻辑：Fr 和 Fo (顶部光流) 交互 ---
        # 1. 计算来自 Fr 的空间注意力
        fr_ap = torch.mean(Fr, dim=1, keepdim=True)     # Average Pooling across channels
        fr_mp = torch.max(Fr, dim=1, keepdim=True)[0] # Max Pooling across channels
        fr_spatial_input = torch.cat([fr_ap, fr_mp], dim=1) # Shape: (B, 2, H, W)
        sa_fr = self.spatial_attention_sigmoid(self.spatial_attention_conv(fr_spatial_input)) # Shape: (B, 1, H, W)

        # 2. 计算来自 Fo 的空间注意力
        fo_ap = torch.mean(Fo, dim=1, keepdim=True)
        fo_mp = torch.max(Fo, dim=1, keepdim=True)[0]
        fo_spatial_input = torch.cat([fo_ap, fo_mp], dim=1) # Shape: (B, 2, H, W)
        sa_fo = self.spatial_attention_sigmoid(self.spatial_attention_conv(fo_spatial_input)) # Shape: (B, 1, H, W)

        # 3. 取两者空间注意力的最大值
        combined_sa_for_fo = torch.max(sa_fr, sa_fo)

        # 4. 空间注意力应用于 Fo
        fo_spatial_attended = Fo * combined_sa_for_fo # 广播机制

        # 5. 应用质量分数 Q1
        #    Fr 和 Fo 交互的最终特征
        final_feat_fr_fo = fo_spatial_attended * Q1 # 广播机制


        # --- 最终融合 ---
        # 三个个分支的特征进行相加
        output_feature = Fr + final_feat_fr_fo + final_feat_fr_for

        return output_feature
    

# ----------------------------------------------ccim模块----------------------------------------------------- 
# 一个辅助函数，用于图像格式和序列格式之间的转换
def image_to_patch(feature): # (B, C, H, W) -> (B, L, C) where L=H*W
    b, c, h, w = feature.shape
    return feature.flatten(2).permute(0, 2, 1)

def patch_to_image(feature, h, w): # (B, L, C) -> (B, C, H, W)
    b, l, c = feature.shape
    return feature.permute(0, 2, 1).view(b, c, h, w)

class ReSoftmax(nn.Module):
    """
    Applies Re-softmax activation: softmax(-x)
    """
    def __init__(self, dim=-1):
        super(ReSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(-x, dim=self.dim)

class MLP(nn.Module):
    """
    A simple Multi-layer Perceptron block.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class StandardCrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0.):
        super(StandardCrossAttentionBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Norm layers for Query, Key, and Value sources
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        
        # Multi-head Attention layer
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=drop,
            batch_first=True
        )
        
        # Post-attention LayerNorm and MLP (Feed-Forward Network)
        self.norm_after_add = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, query_feat, key_feat, value_feat, residual_feat):
        """
        Args:
            query_feat (torch.Tensor): Feature for Query (e.g., F^of or F^rgb).
            key_feat (torch.Tensor): Feature for Key (e.g., F^pre).
            value_feat (torch.Tensor): Feature for Value (e.g., F^pre).
            residual_feat (torch.Tensor): Feature for the residual connection (e.g., F^pre).
        """
        # 1. Pre-normalization and Q, K, V projection
        q = self.norm_q(query_feat)
        k = self.norm_k(key_feat)
        v = self.norm_v(value_feat)
        
        # 2. Multi-head Cross-Attention
        # attn_output shape will be (B, L_q, C), where L_q is the sequence length of query_feat
        attn_output, _ = self.multihead_attn(query=q, key=k, value=v)
        
        # 3. First Residual Connection (Add & Norm)
        # The output of attention is added back to the residual_feat (F^pre)
        # This requires the sequence length of attn_output (from query) to match residual_feat
        assert attn_output.shape == residual_feat.shape, \
               f"Shape mismatch for residual connection: attn_output {attn_output.shape} vs residual {residual_feat.shape}"
        x = residual_feat + attn_output
        x_normed_for_mlp = self.norm_after_add(x)
        
        # 4. Feed-Forward Network with Second Residual Connection
        x_mlp_out = self.mlp(x_normed_for_mlp)
        output = x_normed_for_mlp + x_mlp_out # As per the "Add&Norm" box diagram
        
        return output

class CCIM(nn.Module):
    def __init__(self, channels_of, channels_pre, channels_rgb, out_channels, num_heads=8):
        super(CCIM, self).__init__()
        
        dim = channels_pre # The common dimension is based on F^pre
        assert channels_of == channels_pre == channels_rgb, "All input features must have the same channel dimension."
        assert out_channels == dim, "Output channels must equal input channels for the final addition."

        # Top branch: F^of queries F^pre
        self.cross_attn_of_pre = StandardCrossAttentionBlock(dim=dim, num_heads=num_heads)
        
        # Bottom branch: F^rgb queries F^pre
        self.cross_attn_rgb_pre = StandardCrossAttentionBlock(dim=dim, num_heads=num_heads)

        self.CBR0 = nn.Sequential(
            BasicConv2d(dim * 2, dim, kernel_size=3, padding=1),
            BasicConv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.CBR1 = nn.Sequential(
            BasicConv2d(dim * 2, dim, kernel_size=3, padding=1),
            BasicConv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.CBR2 = nn.Sequential(
            BasicConv2d(dim * 2, dim, kernel_size=3, padding=1),
            BasicConv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        

        
        self.output_channels = out_channels

    def forward(self, feat_of, feat_pre, feat_rgb):
        """
        Args:
            feat_of (torch.Tensor): Feature from OF stream (F^of). Shape: (B, L, C)
            feat_pre (torch.Tensor): Feature from previous stage (F^pre). Shape: (B, L, C)
            feat_rgb (torch.Tensor): Feature from RGB stream (F^rgb). Shape: (B, L, C)
            **IMPORTANT: All features must have the same shape (B, L, C)**
        """

        # --- 准备数据 ---
        # 获取空间维度，用于后续恢复
        B, C, H, W = feat_rgb.shape

        # 转换为序列格式，用于注意力路径
        feat_of_seq = image_to_patch(feat_of)
        feat_pre_seq = image_to_patch(feat_pre)
        feat_rgb_seq = image_to_patch(feat_rgb)

        # --- Top Branch ---
        # F^of is Query
        # F^pre is Key, Value, and the feature for the Residual connection
        attn_out_of_seq  = self.cross_attn_of_pre(
            query_feat=feat_of_seq, 
            key_feat=feat_pre_seq,
            value_feat=feat_pre_seq,
            residual_feat=feat_pre_seq
        )
        attn_out_of_img = patch_to_image(attn_out_of_seq, H, W)

        # --- Bottom Branch ---
        # F^rgb is Query
        # F^pre is Key, Value, and the feature for the Residual connection
        attn_out_rgb_seq = self.cross_attn_rgb_pre(
            query_feat=feat_rgb_seq,
            key_feat=feat_pre_seq,
            value_feat=feat_pre_seq,
            residual_feat=feat_pre_seq
        )
        attn_out_rgb_img = patch_to_image(attn_out_rgb_seq, H, W)

        a = self.CBR0(torch.cat([feat_of, attn_out_of_img], dim=1))
        b = self.CBR1(torch.cat([a, attn_out_rgb_img], dim=1))
        c = self.CBR2(torch.cat([b, feat_rgb], dim=1))
        # --- Final Fusion ---
        # Element-wise addition of the two processed branches
        # Both branches now represent an updated version of F^pre
        output_feature = c

        return output_feature
# ----------------------------------------------CCIM模块-----------------------------------------------------

# ----------------------------------------------GBRM模块-----------------------------------------------------
def DistanceWeight(map_logits):
    """
    Computes a distance transform weight map from a single-channel logit map.
    This operation is NOT differentiable.
    Args:
        map_logits (torch.Tensor): A tensor of shape (B, 1, H, W) representing saliency logits.
    Returns:
        torch.Tensor: A distance transform weight map of shape (B, 1, H, W).
    """
    device = map_logits.device
    
    weights_batch = []
    for i in range(map_logits.shape[0]): # 遍历batch中的每个样本
        # 1. Sigmoid -> CPU -> NumPy -> Binarize
        weight_map = torch.sigmoid(map_logits[i]).cpu().detach().numpy().squeeze()
        weight_map = (weight_map * 255).astype(np.uint8)
        _, binary_map = cv2.threshold(weight_map, 128, 255, cv2.THRESH_BINARY)
        
        # 2. Distance Transform
        dist_transform = cv2.distanceTransform(binary_map, cv2.DIST_L2, 3)
        
        # 3. Normalize
        min_val, max_val = np.min(dist_transform), np.max(dist_transform)
        if max_val - min_val > 1e-8:
            dist_transform = (dist_transform - min_val) / (max_val - min_val)
        else: # Handle cases where the map is all black or all white
            dist_transform = np.zeros_like(dist_transform)

        weights_batch.append(torch.from_numpy(dist_transform))

    # 4. Stack back into a batch tensor and move to the original device
    final_weights = torch.stack(weights_batch, dim=0).unsqueeze(1).float().to(device)
    return final_weights

class GBRM(nn.Module):
    def __init__(self, channels_of, channels_r, out_channels):
        super(GBRM, self).__init__()

        # --- Sub-modules definition ---
        self.conv_bn_single = BasicConv2d(channels_r, channels_r, kernel_size=3, padding=1)
        self.conv_bn = BasicConv2d(channels_of+channels_r, channels_r, kernel_size=3, padding=1)
        
        # DT is now a function, but we need prediction heads to generate its input
        # Prediction head to convert multi-channel feature to single-channel logits
        self.dt_pred_head = nn.Conv2d(channels_of, 1, kernel_size=1)
        # 我们可以让 F^o1 和 F^o2 共享同一个预测头
        
        self.output_channels = out_channels

    def forward(self, feat_of1, feat_r, feat_of2):
        # --- Middle Fr path ---

        fr_feat = self.conv_bn(torch.cat([feat_r, feat_of1], dim=1))
        fr_feat = self.conv_bn(torch.cat([fr_feat, feat_of2], dim=1))
        logits_for_dt_r = self.dt_pred_head(fr_feat)
        dt_map_r = DistanceWeight(logits_for_dt_r) # (B, 1, H, W)
        dt_map_r=dt_map_r.expand(-1, 32, -1, -1)
        o1_fr = self.conv_bn(torch.cat([dt_map_r, feat_of1], dim=1))
        o2_fr = self.conv_bn(torch.cat([dt_map_r, feat_of2], dim=1))

        # --- Left side OF paths ---
        o1_cbr_out = F.relu(self.conv_bn_single(feat_of1), inplace=True)
        o1_gated = o1_cbr_out * feat_r
        # Generate logits from OF feature and then compute DT
        logits_for_dt_o1 = self.dt_pred_head(o1_cbr_out)
        dt_map_o1 = DistanceWeight(logits_for_dt_o1) # (B, 1, H, W)
        dt_map_o1=dt_map_o1.expand(-1, 32, -1, -1)
        # The diagram shows DT being added. It acts as a weight map.
        # We broadcast it to match the channel dimension of o1_gated.
        o1_sum_out = o1_gated + dt_map_o1 
        o1_sum_out=self.conv_bn(torch.cat([o1_sum_out, feat_r], dim=1))
        o1_sum_out=self.conv_bn(torch.cat([o1_sum_out, feat_of2], dim=1))


        o2_cbr_out = F.relu(self.conv_bn_single(feat_of2), inplace=True)
        o2_gated = o2_cbr_out * feat_r
        logits_for_dt_o2 = self.dt_pred_head(o2_cbr_out)
        dt_map_o2 = DistanceWeight(logits_for_dt_o2)
        dt_map_o2=dt_map_o2.expand(-1, 32, -1, -1)
        o2_sum_out = o2_gated + dt_map_o2
        o2_sum_out = self.conv_bn(torch.cat([o2_sum_out, feat_r], dim=1))
        o2_sum_out=self.conv_bn(torch.cat([o1_sum_out, feat_of1], dim=1))

        final_sum=o2_sum_out+o1_fr+o2_fr+o1_sum_out
        output_feature = F.relu(self.conv_bn_single(final_sum), inplace=True)

        return output_feature
# ----------------------------------------------GBRM模块-----------------------------------------------------

# ----------------------------------------------解码器模块-----------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, channels_in_from_deeper, channels_in_from_encoder, channels_out):
        """
        A decoder block that upsamples a deep feature, fuses it with a skip connection 
        from the encoder, and produces an output feature.
        
        Args:
            channels_in_from_deeper (int): Channels of the feature from the deeper layer.
            channels_in_from_encoder (int): Channels of the feature from the encoder skip connection.
            channels_out (int): Desired number of output channels for this block.
        """
        super(DecoderBlock, self).__init__()
        
        # 拼接后的总输入通道数
        concat_channels = channels_in_from_deeper + channels_in_from_encoder
        
        # 使用一个 CBR 块（或几层）来融合拼接后的特征
        # 它接收拼接后的特征，输出 channels_out
        self.fusion_cbr = nn.Sequential(
            BasicConv2d(concat_channels,channels_out,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 也可以用更复杂的结构，例如两层卷积
        # self.fusion_cbr = nn.Sequential(
        #     CBR(concat_channels, out_channels, kernel_size=3, padding=1),
        #     CBR(out_channels, out_channels, kernel_size=3, padding=1)
        # )

    def forward(self, feat_from_deeper, feat_from_encoder):
        """
        Args:
            feat_from_deeper (torch.Tensor): (B, C_deep, H/2, W/2) - Feature from the deeper decoder layer.
            feat_from_encoder (torch.Tensor): (B, C_enc, H, W) - Skip connection from the encoder.
        """
        # 1. Upsample the feature from the deeper layer by a factor of 2
        #    We do this directly in the forward pass using F.interpolate for flexibility.
        feat_from_deeper_upsampled = F.interpolate(
            feat_from_deeper, 
            size=feat_from_encoder.shape[-2:], # Upsample to match the encoder feature size
            mode='bilinear', 
            align_corners=False
        )
        
        # 2. Concatenate the upsampled feature with the encoder feature
        fused_input = torch.cat([feat_from_deeper_upsampled, feat_from_encoder], dim=1)
        
        # 3. Pass through the fusion convolution
        output = self.fusion_cbr(fused_input)
        
        return output
# ----------------------------------------------解码器模块-----------------------------------------------------

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # commented by lzg on 2025.6.5
        # === Encoders ===
        self.rgb_encoder = pvt_v2_b2()  # Independent PVT for RGB
        self.of_encoder_shared = pvt_v2_b2()  # Shared PVT for all 4 Optical Flow streams

        # PVT v2 B2 stage-wise output channels
        pvt_s0_ch, pvt_s1_ch, pvt_s2_ch, pvt_s3_ch = 64, 128, 320, 512
        rfb_out_channels = 32  # All RFBs will output 32 channels
        gbrm_out_channels = 32 # GBRM模块的输出通道数
        ccim_out_channels = 32 # CCIM模块的输出通道数
        fused_out_channels = 32
        decoder_intermediate_channels = 32 # Let's keep the channel count consistent in the decoder

        self.CBR = nn.Sequential(
            BasicConv2d(2*rfb_out_channels,rfb_out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )

        # --- 定义一个可重用的下采样模块 ---
        # 使用 2x2 的最大池化，步长为2，这会将 H 和 W 都减半
        self.downsampler = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义一个可重用的上采样模块
        self.upsampler = nn.Upsample(
            scale_factor=2,       # 放大倍数
            mode='bilinear',      # 使用双线性插值
            align_corners=True    # 一个常用的参数，建议设置为True以保持与旧版本行为一致
        )

        # === RFB Modules for RGB Stream (one for each stage) ===
        self.rgb_rfb_s0 = RFB_modified(pvt_s0_ch, rfb_out_channels)
        self.rgb_rfb_s1 = RFB_modified(pvt_s1_ch, rfb_out_channels)
        self.rgb_rfb_s2 = RFB_modified(pvt_s2_ch, rfb_out_channels)
        self.rgb_rfb_s3 = RFB_modified(pvt_s3_ch, rfb_out_channels)  # This will be used for main fusion

        # === RFB Modules for the Shared OF Stream (one for each stage output of the shared PVT) ===
        # These will process the (B*4, C_stage, H_stage, W_stage) features
        self.of_shared_rfb_s0 = RFB_modified(pvt_s0_ch, rfb_out_channels)
        self.of_shared_rfb_s1 = RFB_modified(pvt_s1_ch, rfb_out_channels)
        self.of_shared_rfb_s2 = RFB_modified(pvt_s2_ch, rfb_out_channels)
        self.of_shared_rfb_s3 = RFB_modified(pvt_s3_ch, rfb_out_channels)  # This will be reshaped and used for main fusion


        # === Auxiliary Prediction Heads for RGB Stream ===
        # 输出1通道logits
        self.aux_pred_head_s0 = nn.Conv2d(rfb_out_channels, 1, kernel_size=1)
        self.aux_pred_head_s1 = nn.Conv2d(rfb_out_channels, 1, kernel_size=1)
        self.aux_pred_head_s2 = nn.Conv2d(rfb_out_channels, 1, kernel_size=1)
        self.aux_pred_head_s3 = nn.Conv2d(rfb_out_channels, 1, kernel_size=1)

        # Upsamplers for auxiliary predictions to match GT sizes for loss calculation
        # (目标尺寸: 64x64, 32x32, 16x16, 8x8)
        # PVT stage0 (s0) is H/4, W/4. If trainsize=256, then 256/4 = 64. No upsampling needed for s0 for aux1.
        # PVT stage1 (s1) is H/8, W/8. If trainsize=256, then 256/8 = 32. No upsampling needed for s1 for aux2.
        # PVT stage2 (s2) is H/16, W/16. If trainsize=256, then 256/16 = 16. No upsampling needed for s2 for aux3.
        # PVT stage3 (s3) is H/32, W/32. If trainsize=256, then 256/32 = 8. No upsampling needed for s3 for aux4.
        # **注意**: 如果你的 trainsize 不同，或者辅助GT尺寸不同，这些上采样器可能需要调整或根本不需要。
        # 为保险起见，我们还是加上，如果输入输出尺寸一样，nn.Upsample也不会做任何事。
        self.upsample_aux_s0_target = nn.Upsample(size=(64, 64), mode='bilinear',
                                                  align_corners=False)  # Target for gts1
        self.upsample_aux_s1_target = nn.Upsample(size=(32, 32), mode='bilinear',
                                                  align_corners=False)  # Target for gts2
        self.upsample_aux_s2_target = nn.Upsample(size=(16, 16), mode='bilinear',
                                                  align_corners=False)  # Target for gts3
        self.upsample_aux_s3_target = nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False)  # Target for gts4

        # ------------------------------------GBRM模块初始化-------------------------------------------------------------------
        # 1. GBRM Modules for OF1 & OF2 chain (shallow to deep)
        self.gbrm_s0_of12 = GBRM(rfb_out_channels, rfb_out_channels, gbrm_out_channels)
        self.gbrm_s1_of12 = GBRM(rfb_out_channels, rfb_out_channels, gbrm_out_channels)

        # 2. GBRM Modules for OF3 & OF4 chain (shallow to deep)
        self.gbrm_s0_of34 = GBRM(rfb_out_channels, rfb_out_channels, gbrm_out_channels)
        self.gbrm_s1_of34 = GBRM(rfb_out_channels, rfb_out_channels, gbrm_out_channels)
        # ------------------------------------GBRM模块初始化-------------------------------------------------------------------
        
        # ------------------------------------DFRM模块初始化--------------------------------------------------------------------------
        self.dfrm_s0_of12 = DFRM(channels=rfb_out_channels)
        self.dfrm_s1_of12 = DFRM(channels=rfb_out_channels)
        self.dfrm_s2_of12 = DFRM(channels=rfb_out_channels)
        self.dfrm_s3_of12 = DFRM(channels=rfb_out_channels)

        self.dfrm_s0_of34 = DFRM(channels=rfb_out_channels)
        self.dfrm_s1_of34 = DFRM(channels=rfb_out_channels)
        self.dfrm_s2_of34 = DFRM(channels=rfb_out_channels)
        self.dfrm_s3_of34 = DFRM(channels=rfb_out_channels)

        self.dfrm0 = DFRM(channels=rfb_out_channels)
        self.dfrm1 = DFRM(channels=rfb_out_channels)
        self.dfrm2 = DFRM(channels=rfb_out_channels)
        self.dfrm3 = DFRM(channels=rfb_out_channels)
        # -------------------------------------DFRM模块初始化-------------------------------------------------------------------------

        # ------------------------------------CCIM模块初始化-------------------------------------------------------------------        
        self.ccim0 =  CCIM(rfb_out_channels, rfb_out_channels, rfb_out_channels, ccim_out_channels)
        self.ccim1 =  CCIM(rfb_out_channels, rfb_out_channels, rfb_out_channels, ccim_out_channels)
        self.ccim2 =  CCIM(rfb_out_channels, rfb_out_channels, rfb_out_channels, ccim_out_channels)
        self.ccim3 =  CCIM(rfb_out_channels, rfb_out_channels, rfb_out_channels, ccim_out_channels)
        
        # ------------------------------------CCIM模块初始化-------------------------------------------------------------------        

        
        # -------------------------------------解码器-------------------------------------------------------------------------


        # --- Final Prediction Layer ---
        # Input is decoder_out_s0 (32ch), which is at H/4 resolution
        # First, upsample by 4 to the original resolution
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        # Then, a final 3x3 convolution to get the 1-channel prediction
        self.final_pred_conv = nn.Conv2d(decoder_intermediate_channels, 1, kernel_size=3, padding=1)
        # -------------------------------------解码器-------------------------------------------------------------------------


        # # commented by lzg on 20250522
        # # focal
        # self.focal_encoder = pvt_v2_b2()
        # self.rfb4 = RFB_modified(512, 32)
        # self.rfb3 = RFB_modified(320, 32)
        # self.rfb2 = RFB_modified(128, 32)
        # self.rfb1 = RFB_modified(64, 32)
        # # self.decoder1 = decoder1(32)
        # # self.decoder2 = decoder2(384)
        # self.transformerdecoder = transfmrerDecoder(6, 4, 32)
        # self.qry = nn.Parameter(torch.zeros(1, 4, 32))
        # self.bn1 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True)
        # self.bn2 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True)
        # self.bn12 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True)
        # conv = nn.Sequential()
        # conv.add_module('conv1', nn.Conv2d(32, 96, 3, 1, 1))   #commented by lzg on 2025.2.22
        # conv.add_module('bn1', self.bn1)
        # conv.add_module('relu1', nn.ReLU(inplace=True))
        # conv.add_module('conv2', nn.Conv2d(96, 1, 3, 1, 1))
        # conv.add_module('bn2', self.bn2)
        # conv.add_module('relu2', nn.ReLU(inplace=True))
        # self.conv = conv
        # self.upsample = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1))
        # self.conv_last = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        # self.upsample0 = nn.Upsample(scale_factor=32, mode='bilinear')
        # self.upsample1 = nn.Upsample(scale_factor=16, mode='bilinear')
        # self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear')
        # self.upsample3 = nn.Upsample(scale_factor=4, mode='bilinear')
        #
        #
        # #rgb
        # self.rgb_encoder = pvt_v2_b2()
        #
        # self.rfb33 = RFB_modified(512, 32)
        # self.rfb22 = RFB_modified(320, 32)
        # self.rfb11 = RFB_modified(128, 32)
        # self.rfb00 = RFB_modified(64, 32)
        # self.rgs = nn.ModuleList()
        #
        # for i in range(4):
        #     self.rgs.append(Rgb_guide_sa())
        #
        # #fuse
        # self.mhsa2 = Block(32, 4)  #num_heads 是一个可以自行设定的超参数， 但需要确保 emb_dim 能够被 num_heads 整除
        # self.mhsa3 = Block(32, 4)


    def forward(self, all_of_input_batched, rgb_input):
        # rgb_input: (B, 3, H, W)
        # all_of_input_batched: (B*4, 3, H, W) <-- 这是train.py转换后的OF输入
        # original_batch_size: B (整数)

        num_of_streams = 4 # Total OF streams
        original_batch_size = rgb_input.shape[0]

        # 1. RGB Encoding and RFB processing for all stages
        rgb_pvt_outputs = self.rgb_encoder(rgb_input)  # List: [s0, s1, s2, s3]
        rgb_feat_s0_rfb = self.rgb_rfb_s0(rgb_pvt_outputs[0])  # (B, 32, H/4, W/4)
        rgb_feat_s1_rfb = self.rgb_rfb_s1(rgb_pvt_outputs[1])  # (B, 32, H/8, W/8)
        rgb_feat_s2_rfb = self.rgb_rfb_s2(rgb_pvt_outputs[2])  # (B, 32, H/16, W/16)
        rgb_feat_s3_rfb = self.rgb_rfb_s3(rgb_pvt_outputs[3])  # (B, 32, H/32, W/32) -> For main fusion

        # ------------------------------------------------分离光流图-----------------------------------------
        # --- 2. Shared OF Encoding, RFB, and Stream Separation (as implemented in previous step) ---
        of_pvt_outputs_batched = self.of_encoder_shared(all_of_input_batched)
        
        of_s0_rfb_batched = self.of_shared_rfb_s0(of_pvt_outputs_batched[0])
        of_s1_rfb_batched = self.of_shared_rfb_s1(of_pvt_outputs_batched[1])
        of_s2_rfb_batched = self.of_shared_rfb_s2(of_pvt_outputs_batched[2])
        of_s3_rfb_batched = self.of_shared_rfb_s3(of_pvt_outputs_batched[3])


        # Helper to separate batched OF features (original_batch_size, num_of_total_streams, C, H, W)
        def separate_streams(batched_feat):
            _, C, H, W = batched_feat.shape
            return batched_feat.view(original_batch_size, num_of_streams, C, H, W)

        of_s0_separated = separate_streams(of_s0_rfb_batched) # (B, 4, 32, H/4, W/4)
        of_s1_separated = separate_streams(of_s1_rfb_batched) # (B, 4, 32, H/8, W/8)
        of_s2_separated = separate_streams(of_s2_rfb_batched) # (B, 4, 32, H/16, W/16)
        of_s3_separated = separate_streams(of_s3_rfb_batched) # (B, 4, 32, H/32, W/32)

        # Assign to individual stream variables for clarity (optional, can use indexing)
        of1_s0, of2_s0, of3_s0, of4_s0 = of_s0_separated.unbind(dim=1) # Each is (B, 32, H/4, W/4)
        of1_s1, of2_s1, of3_s1, of4_s1 = of_s1_separated.unbind(dim=1)
        of1_s2, of2_s2, of3_s2, of4_s2 = of_s2_separated.unbind(dim=1)
        of1_s3, of2_s3, of3_s3, of4_s3 = of_s3_separated.unbind(dim=1)
        # ------------------------------------------------分离光流图-----------------------------------------

        # ------------------------------------------------GBRM模块--------------------------------------------
        # 原代码
        # Stage 0 for OF1 & OF2
        gbrm_out_s0_of12 = self.gbrm_s0_of12(of1_s0, of2_s0, rgb_feat_s0_rfb)     #原代码
        # Stage 1 for OF1 & OF2
        gbrm_out_s1_of12 = self.gbrm_s1_of12(of1_s1, of2_s1, rgb_feat_s1_rfb)     #原代码
        # Stage 2 for OF1 & OF2
        s2_of12 = of1_s2 + of2_s2
        # Stage 3 for OF1 & OF2
        s3_of12 = of1_s3 + of2_s3
        # --- 2. gbrm Chain for OF3 and OF4 (Shallow to Deep) ---
        # Stage 0 for OF3 & OF4
        gbrm_out_s0_of34 = self.gbrm_s0_of34(of3_s0, of4_s0, rgb_feat_s0_rfb)     #原代码
        # Stage 1 for OF3 & OF4
        gbrm_out_s1_of34 = self.gbrm_s1_of34(of3_s1, of4_s1, rgb_feat_s1_rfb)     #原代码
        # Stage 2 for OF3 & OF4
        s2_of34 = of3_s2 + of4_s2
        # Stage 3 for OF3 & OF4
        s3_of34 = of3_s3 + of4_s3
        
        # # 消融实验
        # gbrm_out_s0_of12 = of1_s0 + of2_s0 + rgb_feat_s0_rfb   
        # gbrm_out_s1_of12 = of1_s1 + of2_s1 + rgb_feat_s1_rfb    
        # s2_of12 = of1_s2 + of2_s2
        # s3_of12 = of1_s3 + of2_s3
        # gbrm_out_s0_of34 = of3_s0 + of4_s0 + rgb_feat_s0_rfb     
        # gbrm_out_s1_of34 = of3_s1 + of4_s1 + rgb_feat_s1_rfb   
        # s2_of34 = of3_s2 + of4_s2
        # s3_of34 = of3_s3 + of4_s3
        # ------------------------------------------------GBRM模块--------------------------------------------

        # ---------------------------------------DFRM模块----------------------------------------------------------
        # 原代码
        # OF12
        # 3
        output_dfrm_s3_of12 = self.dfrm_s3_of12(rgb_feat_s3_rfb, s3_of12, s3_of12)    
        output_dfrm_s3_of12_up = F.interpolate(output_dfrm_s3_of12, size=rgb_feat_s2_rfb.shape[-2:], mode='bilinear', align_corners=False)
        # 2
        output_dfrm_s2_of12 = self.dfrm_s2_of12(rgb_feat_s2_rfb, output_dfrm_s3_of12_up, s2_of12)    
        output_dfrm_s2_of12_up = F.interpolate(output_dfrm_s2_of12, size=rgb_feat_s1_rfb.shape[-2:], mode='bilinear', align_corners=False)
        # 1
        output_dfrm_s1_of12 = self.dfrm_s1_of12(rgb_feat_s1_rfb, output_dfrm_s2_of12_up, gbrm_out_s1_of12)    
        output_dfrm_s1_of12_up = F.interpolate(output_dfrm_s1_of12, size=rgb_feat_s0_rfb.shape[-2:], mode='bilinear', align_corners=False)
        # 0
        output_dfrm_s0_of12 = self.dfrm_s0_of12(rgb_feat_s0_rfb, output_dfrm_s1_of12_up, gbrm_out_s0_of12)        

        # OF34
        # 3
        output_dfrm_s3_of34 = self.dfrm_s3_of34(rgb_feat_s3_rfb, s3_of34, s3_of34)    
        output_dfrm_s3_of34_up = F.interpolate(output_dfrm_s3_of34, size=rgb_feat_s2_rfb.shape[-2:], mode='bilinear', align_corners=False)
        # 2
        output_dfrm_s2_of34 = self.dfrm_s2_of34(rgb_feat_s2_rfb, output_dfrm_s3_of34_up, s2_of34)    
        output_dfrm_s2_of34_up = F.interpolate(output_dfrm_s2_of34, size=rgb_feat_s1_rfb.shape[-2:], mode='bilinear', align_corners=False)
        # 1
        output_dfrm_s1_of34 = self.dfrm_s1_of34(rgb_feat_s1_rfb, output_dfrm_s2_of34_up, gbrm_out_s1_of34)    
        output_dfrm_s1_of34_up = F.interpolate(output_dfrm_s1_of34, size=rgb_feat_s0_rfb.shape[-2:], mode='bilinear', align_corners=False)
        # 0
        output_dfrm_s0_of34 = self.dfrm_s0_of34(rgb_feat_s0_rfb, output_dfrm_s1_of34_up, gbrm_out_s0_of34)        

        # # 消融实验
        # # OF12
        # # 3
        # output_dfrm_s3_of12 = rgb_feat_s3_rfb + s3_of12 + s3_of12    
        # output_dfrm_s3_of12_up = F.interpolate(output_dfrm_s3_of12, size=rgb_feat_s2_rfb.shape[-2:], mode='bilinear', align_corners=False)
        # # 2
        # output_dfrm_s2_of12 = rgb_feat_s2_rfb + output_dfrm_s3_of12_up + s2_of12    
        # output_dfrm_s2_of12_up = F.interpolate(output_dfrm_s2_of12, size=rgb_feat_s1_rfb.shape[-2:], mode='bilinear', align_corners=False)
        # # 1
        # output_dfrm_s1_of12 = rgb_feat_s1_rfb + output_dfrm_s2_of12_up + gbrm_out_s1_of12    
        # output_dfrm_s1_of12_up = F.interpolate(output_dfrm_s1_of12, size=rgb_feat_s0_rfb.shape[-2:], mode='bilinear', align_corners=False)
        # # 0
        # output_dfrm_s0_of12 = rgb_feat_s0_rfb + output_dfrm_s1_of12_up + gbrm_out_s0_of12        

        # # OF34
        # # 3
        # output_dfrm_s3_of34 = rgb_feat_s3_rfb + s3_of34 + s3_of34    
        # output_dfrm_s3_of34_up = F.interpolate(output_dfrm_s3_of34, size=rgb_feat_s2_rfb.shape[-2:], mode='bilinear', align_corners=False)
        # # 2
        # output_dfrm_s2_of34 = rgb_feat_s2_rfb + output_dfrm_s3_of34_up + s2_of34    
        # output_dfrm_s2_of34_up = F.interpolate(output_dfrm_s2_of34, size=rgb_feat_s1_rfb.shape[-2:], mode='bilinear', align_corners=False)
        # # 1
        # output_dfrm_s1_of34 = rgb_feat_s1_rfb + output_dfrm_s2_of34_up + gbrm_out_s1_of34    
        # output_dfrm_s1_of34_up = F.interpolate(output_dfrm_s1_of34, size=rgb_feat_s0_rfb.shape[-2:], mode='bilinear', align_corners=False)
        # # 0
        # output_dfrm_s0_of34 = rgb_feat_s0_rfb + output_dfrm_s1_of34_up + gbrm_out_s0_of34

        # --------------------------------------DFRM模块------------------------------------------------------------

        #-------------------------------------------------DFRM相加----------------------------------------------
        dfrm_out_add_s3 = output_dfrm_s3_of12 + output_dfrm_s3_of34
        dfrm_out_add_s2 = output_dfrm_s2_of12 + output_dfrm_s2_of34
        dfrm_out_add_s1 = output_dfrm_s1_of12 + output_dfrm_s1_of34
        dfrm_out_add_s0 = output_dfrm_s0_of12 + output_dfrm_s0_of34
        #-------------------------------------------------DFRM相加----------------------------------------------

        # ------------------------------------------------CCIM模块(解码器)--------------------------------------------
        # 原代码
        ccim_out_s3 = self.ccim3(dfrm_out_add_s3, rgb_feat_s3_rfb, rgb_feat_s3_rfb)
        ccim_out_s3_up = self.upsampler(ccim_out_s3)

        ccim_out_s2 = self.ccim2(dfrm_out_add_s2, ccim_out_s3_up, rgb_feat_s2_rfb)
        ccim_out_s2_up = self.upsampler(ccim_out_s2)

        ccim_out_s1 = self.ccim1(dfrm_out_add_s1, ccim_out_s2_up, rgb_feat_s1_rfb)
        ccim_out_s1_up = self.upsampler(ccim_out_s1)

        ccim_out_s0 = self.ccim0(dfrm_out_add_s0, ccim_out_s1_up, rgb_feat_s0_rfb)

        # # 消融实验
        # ccim_out_s3 = dfrm_out_add_s3 + rgb_feat_s3_rfb + rgb_feat_s3_rfb
        # ccim_out_s3_up = self.upsampler(ccim_out_s3)

        # ccim_out_s2 = dfrm_out_add_s2 + ccim_out_s3_up + rgb_feat_s2_rfb
        # ccim_out_s2_up = self.upsampler(ccim_out_s2)

        # ccim_out_s1 = dfrm_out_add_s1 + ccim_out_s2_up + rgb_feat_s1_rfb
        # ccim_out_s1_up = self.upsampler(ccim_out_s1)

        # ccim_out_s0 = dfrm_out_add_s0 + ccim_out_s1_up + rgb_feat_s0_rfb
        # ------------------------------------------------CCIM模块(解码器)--------------------------------------------

        # ------------------------------------------------最终结果--------------------------------------------
        final_features_full_res = self.final_upsample(ccim_out_s0)
        main_saliency_map_logits = self.final_pred_conv(final_features_full_res)
        # ------------------------------------------------最终结果--------------------------------------------

        # -------------------------------------------------深度监督rgb流的4阶段特征图------------------------------------------
        supervision0 = ccim_out_s0
        supervision1 = ccim_out_s1
        supervision2 = ccim_out_s2
        supervision3 = ccim_out_s3
        # 3. Generate Auxiliary Predictions (assuming based on RGB stream for now, as per __init__)
        # Original loss order: x1 (gts1, H/4), x2 (gts2, H/8), x3 (gts3, H/16), x4 (gts4, H/32)
        aux_pred_s0_logits = self.aux_pred_head_s0(supervision0)  # Corresponds to gts1 (H/4)
        aux_pred_s1_logits = self.aux_pred_head_s1(supervision1)  # Corresponds to gts2 (H/8)
        aux_pred_s2_logits = self.aux_pred_head_s2(supervision2)  # Corresponds to gts3 (H/16)
        aux_pred_s3_logits = self.aux_pred_head_s3(supervision3)  # Corresponds to gts4 (H/32)

        # Ensure auxiliary predictions are at target sizes (can be done here or before loss)
        # The upsample_aux_sX_target modules in __init__ are for this.
        pred_for_gts1 = self.upsample_aux_s0_target(aux_pred_s0_logits)
        pred_for_gts2 = self.upsample_aux_s1_target(aux_pred_s1_logits)
        pred_for_gts3 = self.upsample_aux_s2_target(aux_pred_s2_logits)
        pred_for_gts4 = self.upsample_aux_s3_target(aux_pred_s3_logits)
        # Note: If trainsize=256, and target aux GT sizes are 64,32,16,8, these upsamplers might not change shape.
        # -------------------------------------------------深度监督rgb流的4阶段特征图------------------------------------------


        # --------------------------------------直接融合5流特征----------------------------------------------------------------------------
        # # 5. Fuse features using self.fusion_module (SimpleFusionModule instance)
        # #    Input to SimpleFusionModule is a list containing all features to be concatenated.
        # #    The first feature in the list will be the RGB feature for fusion (rgb_feat_s3_rfb).
        # all_features_for_fusion_input = [rgb_feat_s3_rfb, of1_s3, of2_s3, of3_s3, of4_s3]
        
        # fused_features = self.fusion_module(all_features_for_fusion_input)  # self.fusion_module is SimpleFusionModule
        
        # # 6. Decode to Main Saliency Map (Logits)
        # main_saliency_map_logits = self.main_decoder_module(fused_features)
        # --------------------------------------直接融合5流特征-----------------------------------------------------------------------------------


        # # 看一下rgb流的显著性预测图   2025.6.7
        # rgb_feat_s3_rfb = torch.cat((rgb_feat_s3_rfb, rgb_feat_s3_rfb), dim=1)
        # rgb_feat_s3_rfb = self.main_decoder_module(rgb_feat_s3_rfb)

        # 7. Return predictions in the order expected by train.py's loss calculation
        # Order: pred_for_gts1 (H/4), pred_for_gts2 (H/8), pred_for_gts3 (H/16), pred_for_gts4 (H/32), main_pred (H)
        return pred_for_gts1, pred_for_gts2, pred_for_gts3, pred_for_gts4, main_saliency_map_logits, None, None, None, None  # Placeholders for out_xf, out_xq, sde, fuse_sal if not used


        # # commented bylzg on 20250522
        # #focal
        # ba = x.size()[0]//1
        # x = self.focal_encoder(x)
        # x0f = self.rfb1(x[0])                        # [ba*12, 32, 64, 64]
        # x1f = self.rfb2(x[1])                        # [ba*12, 32, 32, 32]
        # x2f = self.rfb3(x[2])                        # [ba*12, 32, 16, 16]
        # x3f = self.rfb4(x[3])                        # [ba*12, 32, 8, 8]
        # # print(x0f.shape)
        # x0f_re = x0f.reshape(1*ba, 32, 64*64).permute(0, 2, 1)
        # x1f_re = x1f.reshape(1*ba, 32, 32*32).permute(0, 2, 1)
        # x2f_re = x2f.reshape(1*ba, 32, 16*16).permute(0, 2, 1)
        # x3f_re = x3f.reshape(1*ba, 32, 8*8).permute(0, 2, 1)
        #
        # # k_v = self.decoder1(x0f, x1f, x2f, x3f)  # [12*ba, 32, 8, 8]
        # # k_v = k_v.reshape(12*ba, 32, -1).permute(0, 2, 1).contiguous()  # [12*ba, 64, 32]
        # k_v = torch.cat((x0f_re, x1f_re), dim=1)
        # k_v = torch.cat((k_v, x2f_re), dim=1)
        # k_v = torch.cat((k_v, x3f_re), dim=1)
        # # print(k_v.shape)  # [24, 5440, 32]
        #
        # qry = self.qry.repeat(ba, 1, 1)
        # qry = self.transformerdecoder(qry, k_v)  # [24, 16, 32]
        # qry = torch.softmax(torch.cat(torch.chunk(qry.unsqueeze(1), ba, dim=0), dim=1), dim=0)  # [12, 2, 16, 32]
        # # qry = torch.cat(torch.chunk(qry.unsqueeze(1), ba, dim=0), dim=1)  # [12, 2, 16, 32]
        # qry = qry.reshape(1, ba, 2, 2, -1).permute(0, 1, 4, 2, 3).contiguous()  # [12, 2, 32, 4, 4]
        # qry = torch.cat(torch.chunk(qry, ba, dim=1), dim=0).squeeze(1)   # [12*ba, 32, 4, 4]
        #
        # qry0 = self.upsample0(qry)
        # qry1 = self.upsample1(qry)
        # qry2 = self.upsample2(qry)
        # qry3 = self.upsample3(qry)
        #
        #
        # x0q = torch.mul(x0f, qry0)
        # x1q = torch.mul(x1f, qry1)
        # x2q = torch.mul(x2f, qry2)
        # x3q = torch.mul(x3f, qry3)
        #
        # out_xq = x0q
        #
        #
        # x0q_sal = torch.cat(torch.chunk(x0q.unsqueeze(1), ba, dim=0), dim=1)    # [12, ba, 32, 64, 64]
        # x0q_sal = torch.cat(torch.chunk(x0q_sal, 1, dim=0), dim=2).squeeze(0)  # [ba, 384, 64, 6
        # x[0] = x0q_sal
        # x0q_sal = self.conv(x0q_sal)                                            # [ba, 1, 64, 64]
        #
        # x1q_sal = torch.cat(torch.chunk(x1q.unsqueeze(1), ba, dim=0), dim=1)    # [12, ba, 32, 32, 32]
        # x1q_sal = torch.cat(torch.chunk(x1q_sal, 1, dim=0), dim=2).squeeze(0)  # [ba, 384, 32, 32]
        # x[1] = x1q_sal
        # x1q_sal = self.conv(x1q_sal)                                            # [ba, 1, 32, 32]
        #
        # x2q_sal = torch.cat(torch.chunk(x2q.unsqueeze(1), ba, dim=0), dim=1)    # [12, ba, 32, 16, 16]
        # x2q_sal = torch.cat(torch.chunk(x2q_sal, 1, dim=0), dim=2).squeeze(0)  # [ba, 384, 16, 16]
        # x2_a = x2q_sal.reshape(ba, 32, 16*16).permute(0, 2, 1)
        # x[2] = x2q_sal
        # x2q_sal = self.conv(x2q_sal)                                            # [ba, 1, 16, 16]
        #
        # x3q_sal = torch.cat(torch.chunk(x3q.unsqueeze(1), ba, dim=0), dim=1)    # [12, ba, 32, 8, 8]
        # x3q_sal = torch.cat(torch.chunk(x3q_sal, 1, dim=0), dim=2).squeeze(0)  # [ba, 384, 8, 8]
        # x3_a = x3q_sal.reshape(ba, 32, 8 * 8).permute(0, 2, 1)
        # x[3] = x3q_sal
        # x3q_sal = self.conv(x3q_sal)                                            # [ba, 1, 8, 8]
        #
        #
        # #rgb
        # y = self.rgb_encoder(y)
        # y[0] = self.rfb00(y[0])  # [ba, 32, 64, 64]
        # y[1] = self.rfb11(y[1])  # [ba, 32, 32, 32]
        # y[2] = self.rfb22(y[2])  # [ba, 32, 16, 16]
        # y2_a = y[2].reshape(ba, 32, 16*16).permute(0, 2, 1)
        # y[3] = self.rfb33(y[3])  # [ba, 32, 8, 8]
        # y3_a = y[3].reshape(ba, 32, 8*8).permute(0, 2, 1)
        # out_xf = y[0]
        # xy2 = self.mhsa2(x2_a, y2_a)  # [2, 512, 384]
        # xy2_fuse = xy2[:, 0:256, :] + xy2[:, 256:, :]  # [2, 256, 384]
        # xy3 = self.mhsa3(x3_a, y3_a)  # [2, 128, 384]
        # xy3_fuse = xy3[:, 0:64, :] + xy3[:, 64:, :]  # [2, 256, 384]
        #
        #
        # xy2_fuse = xy2_fuse.reshape(ba, 16, 16, -1).permute(0, 3, 1, 2).contiguous()  # [2, 384, 16, 16]
        # xy3_fuse = xy3_fuse.reshape(ba, 8, 8, -1).permute(0, 3, 1, 2).contiguous()  # [2, 384, 8, 8]
        #
        # xy3_fuse = F.interpolate(xy3_fuse, scale_factor=2, mode='bilinear', align_corners=False)
        #
        # xy23 = xy2_fuse + xy3_fuse
        #
        # s = []
        #
        # for i in range(1, -1, -1):
        #     if i == 1:
        #         r = self.rgs[i](y[i], x[i], xy23)
        #     else:
        #         r = self.rgs[i](y[i], x[i], s[0])
        #     s.insert(0, r)
        #
        # #print(s[3].shape, s[2].shape, s[1].shape, s[0].shape)  torch.Size([2, 384, 8, 8]) torch.Size([2, 384, 16, 16]) torch.Size([2, 384, 32, 32]) torch.Size([2, 384, 64, 64])
        #
        # sde = s[0]
        # fuse_sal = self.conv_last(s[0])  # [2, 1, 64, 64]
        # fuse_pred = F.interpolate(fuse_sal, size=(256, 256), mode='bilinear', align_corners=False)  # [2, 1, 256, 256]
        #
        #
        # return x0q_sal, x1q_sal, x2q_sal, x3q_sal, fuse_pred, out_xf, out_xq, sde, fuse_sal


if __name__ == '__main__':
    import torchvision
    from ptflops import get_model_complexity_info
    import time

    from torchstat import stat
    # path = "../config/hrt_base.yaml"
    a = torch.randn(12, 3, 256, 256).cuda() #commented by lzg on 2025.2.22  first param change from 24 to 10
    b = torch.randn(12, 3, 256, 256).cuda()
    # c = torch.randn(1, 1, 352, 352).cuda()
    # config = yaml.load(open(path, "r"),yaml.SafeLoader)['MODEL']['HRT']
    # hr_pth_path = r"E:\ScientificResearch\pre_params\hrt_base.pth"
    # cnn_pth_path = r"D:\tanyacheng\Experiments\pre_trained_params\swin_base_patch4_window7_224_22k.pth"
    # cnn_pth_path = r"E:\ScientificResearch\pre_params\resnet18-5c106cde.pth"
    model = model().cuda()
    # model.initialize_weights()

    # out = model(a, b)

    # stat(model, (b, a))
    # 分析FLOPs
    # flops = FlopCountAnalysis(model, (a, b))
    # print("FLOPs: ", flops.total())
    #
    # # 分析parameters
    # print(parameter_count_table(model))

    # -- coding: utf-8 --


    # model = torchvision.pvt_v2_b2().alexnet(pretrained=False)
    # flops, params = get_model_complexity_info(model, a, as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)

    # params, flops = profile(model, inputs=(b,))
    # params, flops = clever_format([params, flops], "%.2f")
    #
    # print(params, flops)
    # print(out.shape)
    # for x in out:
    #     print(x.shape)


    ###### FPS


    # nums = 710
    # time_s = time.time()
    # for i in range(nums):
    #     _ = model(a, b, c)
    # time_e = time.time()
    # fps = nums / (time_e - time_s)
    # print("FPS: %f" % fps)