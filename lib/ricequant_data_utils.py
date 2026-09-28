"""通用 RiceQuant 数据结构适配函数。"""

import pandas as pd

RQ_DATA_UTILS_VERSION = "0.1.0"


def to_wide_price_panel(raw_price_data, *, value_column=None):
    """
    将 API 返回的价格数据转换为“日期 × 资产”的宽表。

    支持：
    - 已是 DatetimeIndex 的宽表；
    - (日期, 资产) 或 (资产, 日期) 的 MultiIndex 长表；
    - 索引元素为 tuple 的普通 Index 长表。

    参数：
    - raw_price_data：Series 或 DataFrame
    - value_column：长表中存放价格的列名；若只有一列可省略

    返回：
    - DataFrame，索引为标准化交易日期、列为资产代码
    """
    if isinstance(raw_price_data, pd.Series):
        column_name = value_column or raw_price_data.name or "value"
        frame = raw_price_data.to_frame(name=column_name)

    elif isinstance(raw_price_data, pd.DataFrame):
        frame = raw_price_data.copy()

    else:
        raise TypeError(
            "raw_price_data 必须是 pandas Series 或 DataFrame，"
            f"实际为：{type(raw_price_data).__name__}"
        )

    if (
        not isinstance(frame.index, pd.MultiIndex)
        and len(frame.index) > 0
        and isinstance(frame.index[0], tuple)
    ):
        frame.index = pd.MultiIndex.from_tuples(frame.index)

    if isinstance(frame.index, pd.DatetimeIndex):
        wide_panel = frame.copy()

    elif isinstance(frame.index, pd.MultiIndex):
        if frame.index.nlevels != 2:
            raise ValueError(
                "长表价格索引必须恰好包含“日期、资产”两个层级。"
            )

        if value_column in frame.columns:
            value_series = frame[value_column]

        elif frame.shape[1] == 1:
            value_series = frame.iloc[:, 0]

        else:
            raise ValueError(
                "无法确定价格列。请提供 value_column；"
                "当前列为：" + ", ".join(map(str, frame.columns))
            )

        datetime_levels = []

        for level_number in range(frame.index.nlevels):
            level_values = frame.index.get_level_values(level_number)

            converted_values = pd.to_datetime(
                level_values,
                errors="coerce",
            )

            if pd.Series(converted_values).notna().all():
                datetime_levels.append(level_number)

        if len(datetime_levels) != 1:
            raise ValueError(
                "无法唯一识别 MultiIndex 中的日期层级。"
            )

        date_level = datetime_levels[0]
        asset_level = 1 - date_level

        wide_panel = value_series.unstack(level=asset_level)

    else:
        raise ValueError(
            "无法识别价格数据索引结构："
            + type(frame.index).__name__
        )

    wide_panel.index = pd.DatetimeIndex(
        pd.to_datetime(wide_panel.index)
    ).normalize()

    if wide_panel.index.has_duplicates:
        raise ValueError("转换后的价格面板存在重复日期。")

    wide_panel.columns = wide_panel.columns.astype(str)

    return wide_panel.sort_index()
