import asyncio
import time

from veadk import Agent, Runner
from veadk.tools.builtin_tools.vesearch import vesearch
from veadk.tracing.telemetry.exporters.apmplus_exporter import APMPlusExporter
from veadk.tracing.telemetry.opentelemetry_tracer import OpentelemetryTracer

from utils.akshare_stock_complete_apis import  stock_sse_summary, stock_szse_summary, stock_individual_info_em,\
    stock_gsrl_gsdt_em, stock_zh_a_minute, stock_zh_a_spot, stock_zh_a_hist, stock_zh_a_minute, stock_us_hist, stock_hk_hist,\
    stock_board_industry_name_em, stock_board_industry_hist_em, stock_board_industry_hist_min_em, stock_board_concept_name_em,\
    stock_board_concept_cons_em, stock_board_concept_hist_em, index_zh_a_hist
from utils.data_loader import load_deep_research_prompts


exporters = [APMPlusExporter()]
tracer = OpentelemetryTracer()
agent = Agent(name="financial_deep_research",
              instruction='''
                          你是一个财务分析师agent，擅长统计和分析各类股票数据，在收到任务后请按如下要求操作：
                          1. 请根据任务描述中的提示信息，先查询具体的股票代码、股票名称、公告日期，查询时的输入尽量详细 eg:
                          -任务描述：2024 年 3 月，国内某 A 股半导体设备公司公告交付首台 14nm 明场缺陷检测设备，订单金额 1.7 亿元，查询该公司交付公告当日的半导体设备板块涨跌幅，及公司当日成交额
                          -查询输入：2024 年 3 月，国内某 A 股半导体设备公司公告交付首台 14nm 明场缺陷检测设备（1.7 亿元订单），帮我查询公司名称和股票代码，以及具体的公告日期
                          2. 调用相关工具获取数据并分析，给出最终答案
                          ''',
              tools=[vesearch, stock_sse_summary, stock_szse_summary, stock_individual_info_em,
                     stock_gsrl_gsdt_em, stock_zh_a_minute, stock_zh_a_spot, stock_zh_a_hist, stock_zh_a_minute,
                     stock_us_hist, stock_hk_hist, stock_board_industry_name_em, stock_board_industry_hist_em,
                     stock_board_industry_hist_min_em, stock_board_concept_name_em, stock_board_concept_cons_em,
                     stock_board_concept_hist_em, index_zh_a_hist],
              tracers=[tracer])

if __name__ == '__main__':
    session_id_base = "financial_deep_research"
    prompts = load_deep_research_prompts()
    runner = Runner(
        agent=agent
    )
    for i in range(len(prompts)):
        time.sleep(3)
        session_id = f"{session_id_base}_{i+1}"
        response = asyncio.run(runner.run(messages=prompts[i], session_id=session_id, save_tracing_data=True))
        print(response)
        print(f"Tracing file path: {tracer._trace_file_path}")
        dump_path = asyncio.run(runner.save_eval_set(session_id=session_id, eval_set_id="financial_deep_research"))
        print(dump_path)


