#!/usr/bin/env python3
"""
ml-sharp 模型导出到 ExecuTorch 脚本
用于 Android 端侧推理

使用方法:
1. 先安装依赖: pip install -r requirements.txt
2. 安装 ExecuTorch: pip install executorch
3. 运行导出: python export_executorch.py

支持的导出格式:
- ExecuTorch (.pte) - Android 原生支持
- TorchScript (.pt) - 备选方案
- ONNX (.onnx) - 通用格式

作者: Claude
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

# ============================================================================
# 配置
# ============================================================================

MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
MODEL_PATH = Path("./checkpoints/sharp_2572gikvuh.pt")
OUTPUT_DIR = Path("./output")

# 模型内部分辨率 (必须是 384 的倍数)
# 1536 = 4x4 patches of 384x384 (原始设计，最佳质量)
# 768 = 2x2 patches (移动端可用，质量降低)
MOBILE_RESOLUTION = 1536  # 模型架构要求 1536x1536 输入


# ============================================================================
# Step 1: 下载模型
# ============================================================================

def download_model() -> Path | None:
    """下载 SHARP 模型检查点"""
    if MODEL_PATH.exists():
        LOGGER.info(f"[OK] 模型已存在: {MODEL_PATH}")
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"[..] 下载模型中... ({MODEL_URL})")

    try:
        state_dict = torch.hub.load_state_dict_from_url(
            MODEL_URL, model_dir=str(MODEL_PATH.parent), progress=True
        )
        # 保存到本地
        torch.save(state_dict, MODEL_PATH)
        LOGGER.info(f"[OK] 下载完成: {MODEL_PATH}")
        return MODEL_PATH
    except Exception as e:
        LOGGER.warning(f"[WARN] 下载失败: {e}")
        LOGGER.warning("将使用随机初始化的模型进行导出测试")
        return None


# ============================================================================
# Step 2: 加载模型
# ============================================================================

def load_sharp_model(device: torch.device = torch.device("cpu")):
    """加载 SHARP 模型"""
    LOGGER.info("[..] 加载模型...")

    # 添加 src 到路径
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from sharp.models import PredictorParams, create_predictor

    # 创建模型
    model = create_predictor(PredictorParams())

    # 尝试下载/加载预训练权重
    model_path = download_model()
    if model_path is not None and model_path.exists():
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        LOGGER.info("[OK] 预训练模型加载成功")
    else:
        LOGGER.info("[OK] 使用随机初始化模型 (仅用于导出测试)")

    model.eval()
    model.to(device)
    return model


# ============================================================================
# Step 3: 创建可导出的包装模型
# ============================================================================

class SharpMobileWrapper(nn.Module):
    """
    包装 SHARP 模型，使其更容易导出到移动端

    特点:
    - 固定输入分辨率
    - 简化输入参数 (只需图像)
    - 扁平化输出格式 (不使用 NamedTuple)
    """

    def __init__(self, model: nn.Module, resolution: int = 512):
        super().__init__()
        self.model = model
        self.resolution = resolution
        # 默认 disparity factor (基于 512 分辨率)
        self.register_buffer(
            "default_disparity_factor",
            torch.tensor([1.0], dtype=torch.float32)
        )

    def forward(self, image: torch.Tensor) -> Tuple[
        torch.Tensor,  # positions [B, N, 3]
        torch.Tensor,  # scales [B, N, 3]
        torch.Tensor,  # rotations [B, N, 4]
        torch.Tensor,  # colors [B, N, 3]
        torch.Tensor,  # opacities [B, N, 1]
    ]:
        """
        前向推理

        Args:
            image: [B, 3, H, W] 归一化的 RGB 图像 (0-1 范围)

        Returns:
            positions: [B, N, 3] Gaussian 位置
            scales: [B, N, 3] Gaussian 缩放
            rotations: [B, N, 4] Gaussian 旋转四元数
            colors: [B, N, 3] RGB 颜色
            opacities: [B, N, 1] 不透明度
        """
        batch_size = image.shape[0]
        disparity_factor = self.default_disparity_factor.expand(batch_size)

        # 调用原始模型 (不使用 depth alignment)
        gaussians = self.model(image, disparity_factor, depth=None)

        # 返回扁平化的输出
        return (
            gaussians.mean_vectors,
            gaussians.singular_values,
            gaussians.quaternions,
            gaussians.colors,
            gaussians.opacities,
        )


class SharpMobileWrapperDict(nn.Module):
    """
    返回字典格式的包装器 (用于 TorchScript)
    """

    def __init__(self, model: nn.Module, resolution: int = 512):
        super().__init__()
        self.model = model
        self.resolution = resolution
        self.register_buffer(
            "default_disparity_factor",
            torch.tensor([1.0], dtype=torch.float32)
        )

    def forward(self, image: torch.Tensor) -> dict:
        """
        前向推理，返回字典格式

        Args:
            image: [B, 3, H, W] 归一化的 RGB 图像 (0-1 范围)

        Returns:
            dict: 包含 Gaussian 参数的字典
        """
        batch_size = image.shape[0]
        disparity_factor = self.default_disparity_factor.expand(batch_size)

        gaussians = self.model(image, disparity_factor, depth=None)

        return {
            "positions": gaussians.mean_vectors,
            "scales": gaussians.singular_values,
            "rotations": gaussians.quaternions,
            "colors": gaussians.colors,
            "opacities": gaussians.opacities,
        }


# ============================================================================
# Step 4: 导出到 ExecuTorch
# ============================================================================

def export_to_executorch(model: nn.Module, resolution: int = 512) -> Path | None:
    """导出模型到 ExecuTorch 格式 (.pte)"""
    LOGGER.info(f"\n[..] 导出到 ExecuTorch (分辨率: {resolution}x{resolution})...")

    try:
        from executorch.exir import to_edge, EdgeCompileConfig
    except ImportError:
        LOGGER.error("[FAIL] ExecuTorch 未安装")
        LOGGER.error("请运行: pip install executorch")
        return None

    model.eval()

    # 创建示例输入
    example_input = (torch.randn(1, 3, resolution, resolution),)

    # Step 4.1: 使用 torch.export 导出
    LOGGER.info("  -> torch.export...")
    try:
        exported_program = torch.export.export(
            model,
            example_input,
            strict=False  # 允许动态形状
        )
    except Exception as e:
        LOGGER.error(f"  [FAIL] torch.export 失败: {e}")
        LOGGER.info("\n尝试使用 TorchScript 作为备选方案...")
        return None

    # Step 4.2: 转换为 Edge 格式
    LOGGER.info("  -> to_edge...")
    try:
        edge_program = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,  # 跳过某些检查
            )
        )
    except Exception as e:
        LOGGER.error(f"  [FAIL] to_edge 失败: {e}")
        return None

    # Step 4.3: 导出为 .pte 文件
    LOGGER.info("  -> 生成 .pte 文件...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"sharp_mobile_{resolution}.pte"

    try:
        executorch_program = edge_program.to_executorch()
        with open(output_path, "wb") as f:
            f.write(executorch_program.buffer)
        file_size = output_path.stat().st_size / 1024 / 1024
        LOGGER.info(f"[OK] 导出成功: {output_path}")
        LOGGER.info(f"  文件大小: {file_size:.1f} MB")
        return output_path
    except Exception as e:
        LOGGER.error(f"  [FAIL] 导出失败: {e}")
        return None


# ============================================================================
# Step 5: 备选方案 - 导出到 TorchScript
# ============================================================================

def export_to_torchscript(model: nn.Module, resolution: int = 512) -> Path | None:
    """备选方案: 导出到 TorchScript (.pt)"""
    LOGGER.info(f"\n[..] 导出到 TorchScript (分辨率: {resolution}x{resolution})...")

    model.eval()
    example_input = torch.randn(1, 3, resolution, resolution)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 尝试 trace
    try:
        LOGGER.info("  -> torch.jit.trace...")
        with torch.no_grad():
            traced = torch.jit.trace(model, example_input)

        output_path = OUTPUT_DIR / f"sharp_mobile_{resolution}.pt"
        traced.save(str(output_path))
        file_size = output_path.stat().st_size / 1024 / 1024
        LOGGER.info(f"[OK] TorchScript 导出成功: {output_path}")
        LOGGER.info(f"  文件大小: {file_size:.1f} MB")
        return output_path
    except Exception as e:
        LOGGER.error(f"  [FAIL] trace 失败: {e}")

    # 尝试 script
    try:
        LOGGER.info("  -> torch.jit.script...")
        scripted = torch.jit.script(model)
        output_path = OUTPUT_DIR / f"sharp_mobile_{resolution}_scripted.pt"
        scripted.save(str(output_path))
        file_size = output_path.stat().st_size / 1024 / 1024
        LOGGER.info(f"[OK] TorchScript (script) 导出成功: {output_path}")
        LOGGER.info(f"  文件大小: {file_size:.1f} MB")
        return output_path
    except Exception as e:
        LOGGER.error(f"  [FAIL] script 失败: {e}")

    return None


# ============================================================================
# Step 6: 备选方案 - 导出到 ONNX
# ============================================================================

def export_to_onnx(model: nn.Module, resolution: int = 512) -> Path | None:
    """备选方案: 导出到 ONNX 格式"""
    LOGGER.info(f"\n[..] 导出到 ONNX (分辨率: {resolution}x{resolution})...")

    model.eval()
    example_input = torch.randn(1, 3, resolution, resolution)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"sharp_mobile_{resolution}.onnx"

    try:
        LOGGER.info("  -> torch.onnx.export...")
        with torch.no_grad():
            torch.onnx.export(
                model,
                example_input,
                str(output_path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["image"],
                output_names=[
                    "positions",
                    "scales",
                    "rotations",
                    "colors",
                    "opacities"
                ],
                dynamic_axes={
                    "image": {0: "batch_size"},
                    "positions": {0: "batch_size"},
                    "scales": {0: "batch_size"},
                    "rotations": {0: "batch_size"},
                    "colors": {0: "batch_size"},
                    "opacities": {0: "batch_size"},
                }
            )
        file_size = output_path.stat().st_size / 1024 / 1024
        LOGGER.info(f"[OK] ONNX 导出成功: {output_path}")
        LOGGER.info(f"  文件大小: {file_size:.1f} MB")
        return output_path
    except Exception as e:
        LOGGER.error(f"  [FAIL] ONNX 导出失败: {e}")
        return None


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主导出流程"""
    LOGGER.info("=" * 60)
    LOGGER.info("SHARP 模型导出工具 - Android ExecuTorch")
    LOGGER.info("=" * 60)

    # 选择设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        LOGGER.info(f"[INFO] 使用设备: CUDA")
    else:
        device = torch.device("cpu")
        LOGGER.info(f"[INFO] 使用设备: CPU")

    # 加载原始模型
    sharp_model = load_sharp_model(device=torch.device("cpu"))  # 导出时用 CPU

    # 创建移动端包装器
    LOGGER.info(f"\n[..] 创建移动端包装器 (分辨率: {MOBILE_RESOLUTION})...")
    mobile_model = SharpMobileWrapper(sharp_model, resolution=MOBILE_RESOLUTION)
    mobile_model.eval()
    LOGGER.info("[OK] 包装器创建成功")

    # 验证模型可以运行
    LOGGER.info("\n[..] 验证模型推理...")
    try:
        with torch.no_grad():
            test_input = torch.randn(1, 3, MOBILE_RESOLUTION, MOBILE_RESOLUTION)
            outputs = mobile_model(test_input)
            LOGGER.info(f"[OK] 推理成功!")
            LOGGER.info(f"  输出形状:")
            LOGGER.info(f"    - positions:  {outputs[0].shape}")
            LOGGER.info(f"    - scales:     {outputs[1].shape}")
            LOGGER.info(f"    - rotations:  {outputs[2].shape}")
            LOGGER.info(f"    - colors:     {outputs[3].shape}")
            LOGGER.info(f"    - opacities:  {outputs[4].shape}")
    except Exception as e:
        LOGGER.error(f"[FAIL] 推理失败: {e}")
        sys.exit(1)

    # 导出到各种格式
    results = {}

    # 1. 尝试 ExecuTorch
    pte_path = export_to_executorch(mobile_model, MOBILE_RESOLUTION)
    results["ExecuTorch (.pte)"] = pte_path

    # 2. TorchScript 备选
    pt_path = export_to_torchscript(mobile_model, MOBILE_RESOLUTION)
    results["TorchScript (.pt)"] = pt_path

    # 3. ONNX 备选
    onnx_path = export_to_onnx(mobile_model, MOBILE_RESOLUTION)
    results["ONNX (.onnx)"] = onnx_path

    # 打印总结
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("导出结果汇总")
    LOGGER.info("=" * 60)
    for format_name, path in results.items():
        if path:
            LOGGER.info(f"  [OK] {format_name}: {path}")
        else:
            LOGGER.info(f"  [--] {format_name}: 未生成")

    # Android 使用指南
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Android 使用指南")
    LOGGER.info("=" * 60)
    LOGGER.info(f"""
1. ExecuTorch (.pte) - 推荐用于 Android:
   - 将 .pte 文件放入 app/src/main/assets/
   - 使用 ExecuTorch Android SDK 加载:

     Module module = Module.load(assetFilePath("sharp_mobile_{MOBILE_RESOLUTION}.pte"));
     Tensor input = Tensor.fromBlob(imageData, new long[]{{1, 3, {MOBILE_RESOLUTION}, {MOBILE_RESOLUTION}}});
     Tensor[] outputs = module.forward(input);

2. TorchScript (.pt) - 使用 PyTorch Mobile:
   - 添加依赖: implementation 'org.pytorch:pytorch_android_lite:2.1.0'
   - 加载模型:

     Module module = LiteModuleLoader.load(assetFilePath("sharp_mobile_{MOBILE_RESOLUTION}.pt"));
     Tensor input = Tensor.fromBlob(imageData, new long[]{{1, 3, {MOBILE_RESOLUTION}, {MOBILE_RESOLUTION}}});
     Tensor[] outputs = module.forward(IValue.from(input)).toTuple();

3. ONNX (.onnx) - 使用 ONNX Runtime:
   - 添加依赖: implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'
   - 最广泛的设备兼容性
""")

    # 成功导出的数量
    success_count = sum(1 for p in results.values() if p is not None)
    if success_count > 0:
        LOGGER.info(f"\n[OK] 成功导出 {success_count} 个格式!")
        return 0
    else:
        LOGGER.error("\n[FAIL] 所有导出格式都失败了")
        return 1


if __name__ == "__main__":
    sys.exit(main())
