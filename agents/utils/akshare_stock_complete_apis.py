import akshare as ak
import pandas as pd
from typing import Optional, Literal, List, Dict
import logging

logger = logging.getLogger(__name__)


def stock_sse_summary() -> pd.DataFrame:
    """
    上海证券交易所-股票数据总貌接口
    功能描述：获取上海证券交易所最近交易日的股票市场总貌数据，包含全市场股票、科创板、主板的核心统计指标
    接口来源：http://www.sse.com.cn/market/stockdata/statistic/
    数据限制：单次返回最近1个交易日数据，当前交易日数据需交易所收盘后统计（通常16:00后更新）

    输入参数：无（无需传入任何参数，自动获取最近交易日数据）

    输出参数说明：
        - 项目（object）：统计指标名称，包括但不限于"流通股本"、"总市值"、"平均市盈率"、"上市公司"、"上市股票"、"流通市值"、"报告时间"、"总股本"
        - 股票（object）：全市场股票对应指标的统计值，单位说明：流通股本/总股本为"亿股"，市值为"亿元"，市盈率为"倍"，数量为"家/只"
        - 科创板（object）：科创板板块对应指标的统计值，单位与"股票"列一致
        - 主板（object）：主板板块对应指标的统计值，单位与"股票"列一致
    """
    try:
        df = ak.stock_sse_summary()
        return df.to_dict(orient="records")
        
    except Exception as e:
        msg = f"获取上海证券交易所股票数据总貌失败: {e}"
        logger.error(msg)
        return msg


def stock_szse_summary(date: str) -> pd.DataFrame:
    """
    深圳证券交易所-市场总貌-证券类别统计接口
    功能描述：获取深圳证券交易所指定交易日的各证券类别（股票、基金、债券等）统计数据，含数量、成交金额、市值等指标
    接口来源：http://www.szse.cn/market/overview/index.html
    数据限制：单次返回1个指定交易日数据，当前交易日数据需交易所收盘后统计（通常16:00后更新）

    输入参数说明：
        - date（str，必填）：指定交易日，格式为"YYYYMMDD"（如"20240920"），需为深交所已收盘的交易日，不可传入非交易日（周末/节假日）
    输出参数说明：
        - 证券类别（object）：证券类型名称，包括但不限于"股票"、"主板A股"、"创业板A股"、"ETF"、"债券"、"国债"
        - 数量（int64）：该类别证券的挂牌数量，单位为"只"
        - 成交金额（float64）：该类别证券当日成交总金额，单位为"元"
        - 总市值（float64）：该类别证券的总市值，单位为"元"（债券类证券可能返回NaN）
        - 流通市值（float64）：该类别证券的流通市值，单位为"元"（债券类证券可能返回NaN）
    """
    try:
        df = ak.stock_szse_summary(date=date)
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取深圳证券交易所市场总貌-证券类别统计数据失败: {e}"
        logger.error(msg)
        return msg


def stock_individual_info_em(
        symbol: str,
        timeout: Optional[float] = None
) -> List[Dict]:
    """
    东方财富-个股-股票信息查询
    功能描述：获取指定股票代码的个股详细信息（如基本资料、财务指标、市场数据等）
    接口来源：http://quote.eastmoney.com/concept/sh603777.html?from=classic
    限量：单次返回指定股票代码的全量个股信息

    输入参数：
        - symbol (str)：股票代码，必填；格式为纯数字（如"000001"代表平安银行，"603777"代表来伊份）
        - timeout (float)：超时时间，可选；默认None（不设置超时），单位为秒，用于控制接口请求超时时间

    输出参数（返回List[Dict]，每个Dict对应1项个股信息）：
        - item (str)：信息项名称（如"股票名称"、"所属行业"、"最新价"、"市盈率(TTM)"、"总股本"、"每股收益"等）
        - value (str/int/float)：对应信息项的值（如"平安银行"、"银行业"、"10.50"、"6.89"、"1940.62亿"等，值类型随信息项变化）
    """
    # 调用AKShare接口获取个股信息DataFrame
    try:
        df = ak.stock_individual_info_em(
            symbol=symbol,
            timeout=timeout
        )
        # 转换为字典列表，保留item和value字段
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取东方财富-个股-股票信息查询数据失败: {e}"
        logger.error(msg)
        return msg


def stock_gsrl_gsdt_em(date: str) -> List[Dict]:
    """
    东方财富网-数据中心-股市日历-公司动态
    功能描述：获取指定交易日内所有上市公司的动态信息（如业绩预告、股东大会、重大合同等）
    接口来源：https://data.eastmoney.com/gsrl/gsdt.html
    限量：单次返回指定交易日的全量公司动态数据

    输入参数：
        - date (str)：交易日，必填；格式为"YYYYMMDD"（如"20230808"），需传入有效的A股交易日（非交易日返回空数据）

    输出参数（返回List[Dict]，每个Dict对应一条公司动态记录）：
        - 序号：记录排序序号（int）
        - 代码：上市公司股票代码（str，如"600036"）
        - 简称：上市公司股票简称（str，如"招商银行"）
        - 事件类型：公司动态事件类别（str，如"业绩预告"、"股东大会"、"重大合同"等）
        - 具体事项：事件的详细描述（str，如"2023年半年度业绩预告：预计净利润同比增长10%-20%"）
        - 交易日：数据对应的交易日（str，格式"YYYY-MM-DD"，与输入date参数对应）
    """
    # 调用AKShare接口获取指定交易日的公司动态数据
    try:
        df = ak.stock_gsrl_gsdt_em(date=date)
        # 转换为字典列表返回，保持字段与接口输出一致
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取东方财富网-数据中心-股市日历-公司动态数据失败: {e}"
        logger.error(msg)
        return msg


def stock_zh_a_minute(symbol: str, period: str = "1", adjust: str = "qfq") -> pd.DataFrame:
    """
    A股分钟线数据接口
    功能描述：获取A股市场个股的历史分钟级K线数据，支持多时间周期选择，包含复权处理
    接口来源：整合多数据源的A股分钟级交易数据，确保时间精度与数据完整性
    数据限制：返回数据的时间范围受数据源限制，通常覆盖最近1-2年的分钟线数据

    输入参数说明：
        - symbol（str，必填）：股票代码，格式与日线接口一致，即"sh+上交所代码"或"sz+深交所代码"（如"sh600519"、"sz300750"）
        - period（str，可选）：时间周期，默认值"1"，可选值包括"1"（1分钟）、"5"（5分钟）、"15"（15分钟）、"30"（30分钟）、"60"（60分钟）
        - adjust（str，可选）：复权类型，默认值"qfq"，可选值包括"qfq"（前复权）、"hfq"（后复权）、"none"（不复权）
    输出参数说明：
        - datetime（object）：交易时间，格式为"YYYY-MM-DD HH:MM:SS"
        - open（float64）：该分钟周期开盘价，单位为"元"
        - high（float64）：该分钟周期最高价，单位为"元"
        - low（float64）：该分钟周期最低价，单位为"元"
        - close（float64）：该分钟周期收盘价，单位为"元"
        - volume（int64）：该分钟周期成交量，单位为"股"
    """
    try:
        df = ak.stock_zh_a_minute(symbol=symbol, period=period, adjust=adjust)
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取A股分钟线数据失败: {e}"
        logger.error(msg)
        return msg


def stock_zh_a_spot() -> pd.DataFrame:
    """
    A股实时行情接口
    功能描述：获取A股市场所有挂牌交易股票的实时行情快照，包含最新价、涨跌幅、成交量等核心指标
    接口来源：对接实时行情数据源，数据更新频率与交易所行情同步（通常延迟15分钟内）
    数据限制：仅返回当前交易日的实时行情，非交易时间返回最近一个交易日的收盘行情

    输入参数：无（无需传入参数，自动获取全市场A股实时行情）

    输出参数说明：
        - 代码（object）：股票代码，格式为纯数字（如"600000"、"000001"）
        - 名称（object）：股票名称（如"浦发银行"、"平安银行"）
        - 最新价（float64）：当前最新交易价格，单位为"元"
        - 涨跌幅（float64）：当日涨跌幅比例，单位为"%"
        - 涨跌额（float64）：当日涨跌金额，单位为"元"
        - 成交量（float64）：当日累计成交量，单位为"手"（1手=100股）
        - 成交额（float64）：当日累计成交额，单位为"万元"
        - 换手率（float64）：当日换手率比例，单位为"%"
        - 量比（float64）：当日量比指标（当前成交量与近5日平均成交量的比值）
        - 市盈率（float64）：动态市盈率（TTM），单位为"倍"
        - 市净率（float64）：市净率（股价与每股净资产的比值），单位为"倍"
        - 振幅（float64）：当日振幅（（最高价-最低价）/昨日收盘价），单位为"%"
        - 今开（float64）：当日开盘价，单位为"元"
        - 昨收（float64）：昨日收盘价，单位为"元"
        - 最高（float64）：当日最高价，单位为"元"
        - 最低（float64）：当日最低价，单位为"元"
    """
    try:
        df = ak.stock_zh_a_spot()
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取A股实时行情数据失败: {e}"
        logger.error(msg)
        return msg


def stock_zh_a_hist(
        symbol: str,
        period: Literal["daily", "weekly", "monthly"] = "daily",
        start_date: str = "",
        end_date: str = "",
        adjust: Literal["", "qfq", "hfq"] = "qfq"
) -> pd.DataFrame:
    """
    接口名称：A股历史行情数据
    接口来源：新浪财经/东方财富
    功能描述：获取指定股票的历史K线数据，支持日线、周线、月线，可选择复权类型

    输入参数说明：
        - symbol (str, 必填)：股票代码（格式：6位数字，如"600519"，无需前缀sh/sz）
        - period (str, 可选)：数据周期，默认日线（"daily"），支持"weekly"（周线）、"monthly"（月线）
        - start_date (str, 可选)：开始日期（格式：YYYYMMDD，如"20240101"）
        - end_date (str, 可选)：结束日期（格式：YYYYMMDD，如"20240930"）
        - adjust (str, 可选)：复权类型，默认前复权（"qfq"），支持后复权（"hfq"）、不复权（""）

    输出参数-历史行情数据:
        名称	类型	描述
        日期	object	交易日
        股票代码	object	不带市场标识的股票代码
        开盘	float64	开盘价
        收盘	float64	收盘价
        最高	float64	最高价
        最低	float64	最低价
        成交量	int64	注意单位: 手
        成交额	float64	注意单位: 元
        振幅	float64	注意单位: %
        涨跌幅	float64	注意单位: %
        涨跌额	float64	注意单位: 元
        换手率	float64	注意单位: %

    使用示例：
        # 获取贵州茅台2024年全年日线数据（前复权）
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol="600519", start_date="20240101", end_date="20241231")
        print(df.head())
    """
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust=adjust)
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取A股历史行情数据失败: {e}"
        logger.error(msg)
        return msg


def stock_zh_a_minute(
        symbol: str,
        period: Literal["1", "5", "15", "30", "60"] = "1",
) -> pd.DataFrame:
    """
    接口名称：A股分钟线行情数据
    接口来源：新浪财经
    功能描述：获取指定股票的分钟级K线数据，支持1/5/15/30/60分钟周期

    输入参数说明：
        - symbol (str, 必填)：股票代码（格式：sh600519或sz000001）
        - period (str, 可选)：分钟周期，默认"1"分钟

    输出参数说明：
        - time (datetime64[ns])：交易时间
        - open (float)：开盘价（元）
        - high (float)：最高价（元）
        - low (float)：最低价（元）
        - close (float)：收盘价（元）
        - volume (float)：成交量（股）
    """
    try:
        df = ak.stock_zh_a_minute(symbol=symbol, period=period)
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取A股分钟线数据失败: {e}"
        logger.error(msg)
        return msg


def stock_us_hist(
        symbol: str,
        period: str = "daily",
        start_date: str = "",
        end_date: str = "",
        adjust: str = ""
) -> List[Dict]:
    """
    东方财富网-行情-美股-每日行情（历史数据）
    功能描述：获取指定美股上市公司的历史行情数据，支持日/周/月周期及复权类型选择
    接口来源：https://quote.eastmoney.com/us/ENTX.html#fullScreenChart
    限量：单次返回指定股票、周期、复权类型的全量历史数据；需注意复权参数实际生效情况

    输入参数：
        - symbol (str)：美股代码，必填；格式如"106.TTE"（道达尔）、"AAPL"（苹果）；
          可通过调用 ak.stock_us_spot_em() 获取所有美股代码（取返回DataFrame的"代码"字段）
        - period (str)：数据周期，默认"daily"；可选值：{"daily": 日K, "weekly": 周K, "monthly": 月K}
        - start_date (str)：开始日期，可选；格式"YYYYMMDD"（如"20210101"），空值则默认获取最早可查数据
        - end_date (str)：结束日期，可选；格式"YYYYMMDD"（如"20210601"），空值则默认获取最新数据
        - adjust (str)：复权类型，默认""（不复权）；可选值：{"": 不复权, "qfq": 前复权, "hfq": 后复权}

    输出参数（返回List[Dict]，每个Dict对应一个周期的历史行情）：
        - 日期：历史数据日期（str，格式"YYYY-MM-DD"）
        - 开盘：该周期开盘价（float，单位：美元）
        - 收盘：该周期收盘价（float，单位：美元；复权类型生效时为复权后价格）
        - 最高：该周期最高价（float，单位：美元）
        - 最低：该周期最低价（float，单位：美元）
        - 成交量：该周期成交量（int，单位：股）
        - 成交额：该周期成交额（float，单位：美元）
        - 振幅：该周期振幅（float，单位：%）
        - 涨跌幅：该周期涨跌幅（float，单位：%）
        - 涨跌额：该周期价格涨跌差额（float，单位：美元）
        - 换手率：该周期换手率（float，单位：%）
    """
    # 调用AKShare接口获取美股历史行情DataFrame
    try:
        df = ak.stock_us_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )
        # 转换为字典列表返回，保留所有输出字段
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取美股历史行情数据失败: {e}"
        logger.error(msg)
        return msg


def stock_hk_hist(
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: Optional[str] = ""
) -> List[Dict]:
    """
    香港股市-历史交易数据接口（AKShare封装）
    功能描述：获取港股标的（指数/个股）日度历史交易数据，支持复权选择

    输入参数：
        - symbol (str)：标的代码，指数用代码（如"HSI"），个股用"代码"（如"00700"）
        - start_date (str)：开始日期，格式"YYYY-MM-DD"（如"2024-01-01"）
        - end_date (str)：结束日期，格式"YYYY-MM-DD"（如"2024-05-31"）
        - adjust (str)：复权类型，默认""（不复权），可选"qfq"（前复权）、"hfq"（后复权）

    输出参数（返回List[Dict]，每个Dict对应1个交易日）：
        - 日期：交易日期（YYYY-MM-DD）
        - 开盘价：当日开盘价（港元，指数无单位）
        - 最高价：当日最高价（港元，指数无单位）
        - 最低价：当日最低价（港元，指数无单位）
        - 收盘价：当日收盘价（复权后价格，港元，指数无单位）
        - 成交量：当日成交量（个股单位"股"，指数以实际返回为准）
        - 成交额：当日成交额（港元）
        - 涨跌幅：当日涨跌幅（%）
    """
    # 调用AKShare接口获取原始数据
    try:
        df = ak.stock_hk_hist(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )
        # 转换为字典列表返回（便于后续处理）
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"Failed to get stock hk hist data for symbol {symbol}, start_date {start_date}, end_date {end_date}, adjust {adjust}. Error: {e}"
        logger.error(msg)
        return msg


def stock_board_industry_name_em() -> List[Dict]:
    """
    东方财富-沪深京板块-行业板块列表
    功能描述：获取当前时刻所有行业板块的实时行情数据，包含板块基本信息及行情指标

    输入参数：无

    输出参数（返回List[Dict]，每个Dict对应一个行业板块）：
        - 排名：板块排名序号（int）
        - 板块名称：行业板块名称（str）
        - 板块代码：行业板块代码（str）
        - 最新价：板块当前最新价格（float）
        - 涨跌额：板块价格涨跌额（float）
        - 涨跌幅：板块价格涨跌幅（float，单位：%）
        - 总市值：板块总市值（int）
        - 换手率：板块换手率（float，单位：%）
        - 上涨家数：板块内上涨的股票数量（int）
        - 下跌家数：板块内下跌的股票数量（int）
        - 领涨股票：板块内领涨的股票名称（str）
        - 领涨股票-涨跌幅：领涨股票的涨跌幅（float，单位：%）
    """
    # 调用AKShare接口获取行业板块数据
    try:
        df = ak.stock_board_industry_name_em()
        # 转换为字典列表返回
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取东方财富行业板块列表数据失败: {e}"
        logger.error(msg)
        return msg


def stock_board_industry_hist_em(
        symbol: str,
        start_date: str,
        end_date: str,
        period: str = "日k",
        adjust: str = ""
) -> List[Dict]:
    """
    东方财富-沪深板块-行业板块-历史行情数据（日频/周频/月频）
    功能描述：获取东方财富行业板块的历史行情数据，支持不同周期和复权类型

    输入参数：
        - symbol (str)：行业板块名称，如"小金属"；可通过ak.stock_board_industry_name_em()获取所有行业
        - start_date (str)：开始日期，格式"YYYYMMDD"（如"20211201"）
        - end_date (str)：结束日期，格式"YYYYMMDD"（如"20220401"）
        - period (str)：周期选择，默认"日k"；可选值{"日k", "周k", "月k"}
        - adjust (str)：复权类型，默认""（不复权）；可选值{"": 不复权, "qfq": 前复权, "hfq": 后复权}

    输出参数（返回List[Dict]，每个Dict对应一条记录）：
        - 日期：交易日期（格式YYYY-MM-DD）
        - 开盘：开盘价
        - 收盘：收盘价
        - 最高：最高价
        - 最低：最低价
        - 涨跌幅：涨跌幅（单位%）
        - 涨跌额：涨跌额
        - 成交量：成交量
        - 成交额：成交额
        - 振幅：振幅（单位%）
        - 换手率：换手率（单位%）
    """
    try:
        df = ak.stock_board_industry_hist_em(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            period=period,
            adjust=adjust
        )
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取东方财富行业板块历史行情数据失败: {e}"
        logger.error(msg)
        return msg


def stock_board_industry_hist_min_em(
        symbol: str,
        period: str = "1"
) -> List[Dict]:
    """
    东方财富-沪深板块-行业板块-分时历史行情数据
    功能描述：获取东方财富行业板块的分时行情数据，支持不同分钟周期

    输入参数：
        - symbol (str)：行业板块名称，如"小金属"
        - period (str)：时间周期（分钟），默认"1"；可选值{"1", "5", "15", "30", "60"}

    输出参数（返回List[Dict]，每个Dict对应一条分时记录）：
        - 日期时间：分时数据的时间点（格式YYYY-MM-DD HH:MM）
        - 开盘：该时段开盘价
        - 收盘：该时段收盘价
        - 最高：该时段最高价
        - 最低：该时段最低价
        - 成交量：该时段成交量
        - 成交额：该时段成交额
        - 最新价：该时段最新价
    """
    try:
        df = ak.stock_board_industry_hist_min_em(
            symbol=symbol,
            period=period
        )
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取东方财富行业板块分时历史行情数据失败: {e}"
        logger.error(msg)
        return msg


def stock_board_concept_name_em() -> List[Dict]:
    """
    东方财富网-行情中心-沪深京板块-概念板块列表
    功能描述：获取当前时刻所有概念板块的实时行情数据，包含板块基本信息及行情指标
    接口来源：https://quote.eastmoney.com/center/boardlist.html#concept_board
    限量：单次返回全量概念板块实时数据

    输入参数：无

    输出参数（返回List[Dict]，每个Dict对应一个概念板块）：
        - 排名：板块实时排名序号（int）
        - 板块名称：概念板块名称（str，如"融资融券"）
        - 板块代码：概念板块代码（str，如"BK0655"）
        - 最新价：板块当前最新行情价格（float）
        - 涨跌额：板块价格涨跌差额（float）
        - 涨跌幅：板块价格涨跌幅（float，单位：%）
        - 总市值：板块整体总市值（int）
        - 换手率：板块实时换手率（float，单位：%）
        - 上涨家数：板块内上涨股票数量（int）
        - 下跌家数：板块内下跌股票数量（int）
        - 领涨股票：板块内领涨股票名称（str）
        - 领涨股票-涨跌幅：领涨股票的实时涨跌幅（float，单位：%）
    """
    try:
        df = ak.stock_board_concept_name_em()
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取东方财富概念板块列表数据失败: {e}"
        logger.error(msg)
        return msg


def stock_board_concept_cons_em(symbol: str) -> List[Dict]:
    """
    东方财富-沪深板块-概念板块-板块成份股
    功能描述：获取指定概念板块的所有成份股实时行情数据
    接口来源：http://quote.eastmoney.com/center/boardlist.html#boards-BK06551
    限量：单次返回指定板块全量成份股实时数据

    输入参数：
        - symbol (str)：概念板块标识，支持传入"板块名称"（如"融资融券"）或"板块代码"（如"BK0655"）；
          可通过调用 stock_board_concept_name_em() 获取所有概念板块的名称/代码

    输出参数（返回List[Dict]，每个Dict对应一只成份股）：
        - 序号：成份股在板块内的排序（int）
        - 代码：成份股股票代码（str）
        - 名称：成份股股票名称（str）
        - 最新价：成份股实时最新价（float）
        - 涨跌幅：成份股实时涨跌幅（float，单位：%）
        - 涨跌额：成份股价格涨跌差额（float）
        - 成交量：成份股实时成交量（float，单位：手）
        - 成交额：成份股实时成交额（float）
        - 振幅：成份股实时振幅（float，单位：%）
        - 最高：成份股实时最高价（float）
        - 最低：成份股实时最低价（float）
        - 今开：成份股当日开盘价（float）
        - 昨收：成份股昨日收盘价（float）
        - 换手率：成份股实时换手率（float，单位：%）
        - 市盈率-动态：成份股动态市盈率（float）
        - 市净率：成份股市净率（float）
    """
    try:
        df = ak.stock_board_concept_cons_em(symbol=symbol)
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取东方财富概念板块成份股数据失败: {e}"
        logger.error(msg)
        return msg


def stock_board_concept_hist_em(
        symbol: str,
        period: str = "daily",
        start_date: str = "",
        end_date: str = "",
        adjust: str = ""
) -> List[Dict]:
    """
    东方财富-沪深板块-概念板块-历史行情数据
    功能描述：获取指定概念板块的历史行情数据，支持不同周期和复权类型
    接口来源：http://quote.eastmoney.com/bk/90.BK0715.html
    限量：单次返回指定板块、周期、时间范围的全量历史数据

    输入参数：
        - symbol (str)：概念板块标识，支持"板块名称"（如"绿色电力"）或"板块代码"（如"BK0715"）；
          可通过 stock_board_concept_name_em() 获取
        - period (str)：数据周期，默认"daily"；可选值：{"daily": 日K, "weekly": 周K, "monthly": 月K}
        - start_date (str)：开始日期，格式"YYYYMMDD"（如"20220101"），空值则默认获取最早可查数据
        - end_date (str)：结束日期，格式"YYYYMMDD"（如"20221128"），空值则默认获取最新数据
        - adjust (str)：复权类型，默认""（不复权）；可选值：{"": 不复权, "qfq": 前复权, "hfq": 后复权}

    输出参数（返回List[Dict]，每个Dict对应一个周期的历史数据）：
        - 日期：历史数据日期（str，格式"YYYY-MM-DD"）
        - 开盘：该周期开盘价（float）
        - 收盘：该周期收盘价（float，复权类型生效时为复权后价格）
        - 最高：该周期最高价（float）
        - 最低：该周期最低价（float）
        - 涨跌幅：该周期涨跌幅（float，单位：%）
        - 涨跌额：该周期价格涨跌差额（float）
        - 成交量：该周期成交量（int）
        - 成交额：该周期成交额（float）
        - 振幅：该周期振幅（float，单位：%）
        - 换手率：该周期换手率（float，单位：%）
    """
    try:
        df = ak.stock_board_concept_hist_em(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取东方财富概念板块历史行情数据失败: {e}"
        logger.error(msg)
        return msg


def index_zh_a_hist(
        symbol: str,
        period: str = "daily",
        start_date: str = "19700101",
        end_date: str = "22220101"
) -> List[Dict]:
    """
    东方财富网-中国股票指数-历史行情数据
    功能描述：获取指定中国A股指数（如上证50、创业板指）的历史行情数据，支持日/周/月周期及自定义时间范围
    接口来源：http://quote.eastmoney.com/center/hszs.html
    限量：单次返回指定指数、周期下，从start_date到end_date的近期历史数据（非全量，随时间范围动态调整）

    输入参数：
        - symbol (str)：指数代码，必填；无需加市场标识，直接传纯数字代码（如"000016"代表上证50，"399006"代表创业板指）；
          可通过东方财富网指数页面（http://quote.eastmoney.com/center/hszs.html）查询目标指数代码
        - period (str)：数据周期，默认"daily"；可选值：{"daily": 日K, "weekly": 周K, "monthly": 月K}
        - start_date (str)：开始日期，默认"19700101"（最早可查日期）；格式"YYYYMMDD"（如"20200101"）
        - end_date (str)：结束日期，默认"22220101"（未来远期日期，实际返回最新数据）；格式"YYYYMMDD"（如"20241231"）

    输出参数（返回List[Dict]，每个Dict对应一个周期的指数行情）：
        - 日期：交易日期（str，格式"YYYY-MM-DD"，仅包含指数交易日）
        - 开盘：该周期指数开盘价（float）
        - 收盘：该周期指数收盘价（float）
        - 最高：该周期指数最高价（float）
        - 最低：该周期指数最低价（float）
        - 成交量：该周期成交总量（int，单位：手）
        - 成交额：该周期成交总金额（float，单位：元）
        - 振幅：该周期指数振幅（float，单位：%）
        - 涨跌幅：该周期指数涨跌幅（float，单位：%，相对前一周期收盘价）
        - 涨跌额：该周期指数涨跌差额（float，单位：元）
        - 换手率：该周期指数换手率（float，单位：%）
    """
    try:
        # 调用AKShare接口获取A股指数历史行情DataFrame
        df = ak.index_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date
        )
        # 转换为字典列表返回，保留所有输出字段
        return df.to_dict(orient="records")
    except Exception as e:
        msg = f"获取东方财富中国股票指数历史行情数据失败: {e}"
        logger.error(msg)
        return msg


if __name__ == '__main__':
    # print(stock_hk_hist(symbol='01788', start_date='2025-06-24', end_date='2025-06-24'))
    # print(stock_zh_a_hist(symbol="300601", period="daily", start_date="20200529", end_date="20200529", adjust="qfq"))
    # print(stock_board_industry_hist_em(symbol="化学原料", start_date="20230707", end_date="20230707", period="日k", adjust=""))
    # print(stock_individual_info_em(symbol="601216"))
    # #print(stock_zh_a_daily(symbol='881124', start_date='20250624', end_date='20250624'))
    # print(stock_szse_summary(date="20230828"))
    # print(stock_board_industry_hist_em(symbol="证券", start_date="20230828", end_date="20230828", period="日k", adjust=""))
    # print(stock_zh_a_hist(symbol="601398", period="daily", start_date="20230828", end_date="20230828", adjust=""))
    # print(stock_zh_a_hist(symbol="688086", period="daily", start_date="20230701", end_date="20230706", adjust=""))
    # print(stock_board_concept_hist_em(symbol="退市股", start_date="20230520", end_date="20230706", period="日k", adjust=""))
    # print(stock_gsrl_gsdt_em(date="20230707"))
    # print(stock_board_industry_name_em())
    stock_us_spot_em_df = ak.stock_us_spot_em()
    print(stock_us_spot_em_df)
