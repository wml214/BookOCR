"""图书盘点系统的 Streamlit 应用入口。

当前文件只负责提供工程骨架阶段的启动页面。后续业务逻辑必须放入
``src/book_inventory`` 下对应模块，避免把模型推理、OCR、去重和界面代码
全部堆积在一个文件中。
"""

from __future__ import annotations


def main() -> None:
    """启动 Streamlit 页面。

    Streamlit 仅在函数执行时导入，这样在尚未安装界面依赖的环境中，
    其他数据处理或训练脚本仍然可以导入项目包，而不会立即报错。
    """

    import streamlit as st

    # 页面级配置必须在任何其他 Streamlit 命令之前执行。
    st.set_page_config(
        page_title="基于视觉的图书盘点系统",
        page_icon="📚",
        layout="wide",
    )

    st.title("基于视觉的图书盘点系统")
    st.info(
        "项目骨架已建立。后续将依次接入书脊分割、OCR、馆藏匹配、"
        "跨图去重和盘点结果导出功能。"
    )

    # 使用三列展示当前开发阶段，后续可替换为真实运行指标。
    col_data, col_model, col_app = st.columns(3)
    col_data.metric("现有原始图像", "102 张")
    col_model.metric("计划分割模型", "YOLOv8n-seg")
    col_app.metric("当前阶段", "工程骨架")


if __name__ == "__main__":
    main()

