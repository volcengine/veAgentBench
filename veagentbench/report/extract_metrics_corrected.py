#!/usr/bin/env python3
## Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http:##www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

"""
修正版的Metrics提取工具
正确提取每个test case对应的metricsData里面的score和reason等字段
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import argparse

def extract_metrics_data(test_run_file: str) -> List[Dict[str, Any]]:
    """
    从测试运行文件中提取每个test case的metrics数据
    
    Args:
        test_run_file: 测试运行JSON文件路径
        
    Returns:
        包含所有test case metrics数据的列表
    """
    with open(test_run_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = data['testRunData']['testCases']
    extracted_data = []
    
    for i, test_case in enumerate(test_cases):
        case_data = {
            'case_id': i,
            'case_name': test_case.get('name', f'test_case_{i}'),
            'input': test_case.get('input', ''),
            'actual_output': test_case.get('actualOutput', ''),
            'expected_output': test_case.get('expectedOutput', ''),
            'success': test_case.get('success', False),
            'run_duration': test_case.get('runDuration', 0)
        }
        
        # 提取metricsData中的每个指标
        metrics_data = test_case.get('metricsData', [])
        
        for metric in metrics_data:
            metric_name = metric.get('name', 'Unknown')
            # 清理指标名称，用作列名
            clean_name = metric_name.lower().replace(' ', '_').replace('-', '_')
            
            # 添加指标的各个字段
            case_data[f'{clean_name}_score'] = metric.get('score', None)
            case_data[f'{clean_name}_reason'] = metric.get('reason', '')
            case_data[f'{clean_name}_success'] = metric.get('success', False)
            case_data[f'{clean_name}_threshold'] = metric.get('threshold', None)
            case_data[f'{clean_name}_strict_mode'] = metric.get('strictMode', False)
            case_data[f'{clean_name}_evaluation_model'] = metric.get('evaluationModel', '')
            
            # 提取拆解维度分数（score_breakdown / breakdown / dimension_scores）
            breakdown = metric.get('score_breakdown')
            if breakdown is None:
                breakdown = metric.get('breakdown')
            if breakdown is None:
                breakdown = metric.get('dimension_scores')
            if isinstance(breakdown, dict):
                for dim_key, dim_val in breakdown.items():
                    dim_clean = str(dim_key).lower().replace(' ', '_').replace('-', '_')
                    case_data[f'{clean_name}_breakdown_{dim_clean}'] = dim_val
        
        # 预期工具调用（expected_tool_calls），兼容两种位置
        expected_calls = test_case.get('expectedToolCalls')
        if expected_calls is None:
            execution_data = test_case.get('executionData') or {}
            expected_calls = execution_data.get('expectedToolCalls')
        # 以JSON字符串形式保存到详细表，便于查看
        try:
            case_data['expected_tool_calls'] = json.dumps(expected_calls, ensure_ascii=False) if expected_calls is not None else ''
        except Exception:
            case_data['expected_tool_calls'] = str(expected_calls) if expected_calls is not None else ''
        
        extracted_data.append(case_data)
    
    return extracted_data

def create_metrics_summary(extracted_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    创建metrics汇总统计
    
    Args:
        extracted_data: 提取的metrics数据
        
    Returns:
        汇总统计信息
    """
    if not extracted_data:
        return {}
    
    # 找出所有的指标名称
    metric_names = set()
    for case in extracted_data:
        for key in case.keys():
            if key.endswith('_score'):
                metric_name = key[:-6]  # 移除'_score'后缀
                metric_names.add(metric_name)
    
    summary = {
        'total_cases': len(extracted_data),
        'overall_success_rate': sum(1 for case in extracted_data if case['success']) / len(extracted_data),
        'metrics_summary': {}
    }
    
    # 为每个指标计算统计信息
    for metric_name in metric_names:
        score_key = f'{metric_name}_score'
        success_key = f'{metric_name}_success'
        
        scores = [case[score_key] for case in extracted_data if case.get(score_key) is not None]
        successes = [case[success_key] for case in extracted_data if success_key in case]
        
        if scores:
            summary['metrics_summary'][metric_name] = {
                'avg_score': sum(scores) / len(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'success_rate': sum(successes) / len(successes) if successes else 0,
                'total_evaluations': len(scores)
            }
    
    return summary

def save_to_formats(extracted_data: List[Dict[str, Any]], summary: Dict[str, Any], output_prefix: str):
    """
    保存数据到多种格式
    
    Args:
        extracted_data: 提取的数据
        summary: 汇总统计
        output_prefix: 输出文件前缀
    """
    # 保存为CSV
    df = pd.DataFrame(extracted_data)
    csv_file = f"{output_prefix}_detailed.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"✅ 详细数据已保存到: {csv_file}")
    
    # 保存汇总为JSON
    summary_file = f"{output_prefix}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✅ 汇总统计已保存到: {summary_file}")
    
    # 创建简化的metrics表格
    metrics_rows = []
    for case in extracted_data:
        row = {
            'case_id': case['case_id'],
            'case_name': case['case_name'],
            'overall_success': case['success']
        }
        
        # 添加每个指标的score与拆解维度
        for key, value in case.items():
            if key.endswith('_score'):
                metric_name = key[:-6]
                row[f'{metric_name}_score'] = value
                row[f'{metric_name}_success'] = case.get(f'{metric_name}_success', False)
            # 将拆解维度分数也加入简化表
            if '_breakdown_' in key:
                row[key] = value
        
        metrics_rows.append(row)
    
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_file = f"{output_prefix}_metrics_only.csv"
    metrics_df.to_csv(metrics_file, index=False, encoding='utf-8')
    print(f"✅ 指标数据已保存到: {metrics_file}")
    
    # 生成HTML报告，友好展示总指标与拆解维度
    try:
        html_file = f"{output_prefix}_report.html"
        
        # 汇总表（metrics_summary）转为DataFrame
        metrics_summary = summary.get('metrics_summary', {})
        summary_rows = []
        for name, stats in metrics_summary.items():
            row = {'metric': name}
            row.update(stats)
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        
        # HTML内容构建
        html_parts = []
        html_parts.append("<html><head><meta charset='utf-8'><title>Metrics Report</title>")
        html_parts.append("""
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Microsoft Yahei", sans-serif; margin: 24px; }
h1, h2, h3 { color: #222; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th, td { border: 1px solid #ddd; padding: 8px; font-size: 13px; }
th { background: #f6f8fa; text-align: left; }
code, pre { background: #f6f8fa; padding: 8px; border-radius: 6px; }
.small { color: #666; font-size: 12px; }
.kpi { display: flex; gap: 16px; margin: 8px 0 16px; }
.kpi .card { background: #fafafa; border: 1px solid #eee; padding: 12px 16px; border-radius: 8px; }
.section { margin-top: 20px; }
</style>
</head><body>
""")
        html_parts.append("<h1>Metrics Report / 指标评估报告</h1>")
        html_parts.append(f"<div class='kpi'><div class='card'><b>Total cases</b><br>{summary.get('total_cases', 0)}</div>")
        overall_rate = summary.get('overall_success_rate', 0)
        html_parts.append(f"<div class='card'><b>Overall success rate</b><br>{overall_rate:.2%}</div></div>")
        
        # 汇总表
        html_parts.append("<div class='section'><h2>Metrics Summary / 指标汇总</h2>")
        if not summary_df.empty:
            html_parts.append(summary_df.to_html(index=False))
        else:
            html_parts.append("<div class='small'>No metrics summary.</div>")
        html_parts.append("</div>")
        
        # 简化指标表（含拆解维度）
        html_parts.append("<div class='section'><h2>Metrics (Scores & Breakdowns) / 指标分数与拆解维度</h2>")
        html_parts.append(metrics_df.to_html(index=False))
        html_parts.append("</div>")
        
        # 详细用例表（含 expected_tool_calls 预览）
        html_parts.append("<div class='section'><h2>Detailed Cases / 详细用例</h2>")
        detail_df = df.copy()
        def _shorten(x):
            try:
                s = str(x)
                return s if len(s) <= 300 else (s[:300] + "...(truncated)")
            except Exception:
                return x
        if 'expected_tool_calls' in detail_df.columns:
            detail_df['expected_tool_calls_preview'] = detail_df['expected_tool_calls'].apply(_shorten)
        html_parts.append(detail_df.to_html(index=False))
        html_parts.append("</div>")
        
        html_parts.append("<hr><div class='small'>Generated by extract_metrics_corrected.py</div></body></html>")
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write("".join(html_parts))
        print(f"✅ HTML 报告已生成: {html_file}")
    except Exception as e:
        print(f"⚠️ 生成HTML报告失败: {e}")

def print_metrics_overview(summary: Dict[str, Any]):
    """
    打印metrics概览
    
    Args:
        summary: 汇总统计信息
    """
    print("\n" + "="*60)
    print("📊 METRICS 数据概览")
    print("="*60)
    
    print(f"📋 总测试案例数: {summary['total_cases']}")
    print(f"✅ 整体成功率: {summary['overall_success_rate']:.2%}")
    
    print("\n📈 各指标统计:")
    print("-" * 60)
    
    for metric_name, stats in summary['metrics_summary'].items():
        print(f"\n🔹 {metric_name.replace('_', ' ').title()}")
        print(f"   平均分数: {stats['avg_score']:.3f}")
        print(f"   分数范围: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
        print(f"   成功率: {stats['success_rate']:.2%}")
        print(f"   评估次数: {stats['total_evaluations']}")

def main():
    parser = argparse.ArgumentParser(description='提取测试案例的metrics数据')
    parser.add_argument('--input', '-i', 
                       default='.deepeval/.latest_test_run.json',
                       help='输入的测试运行JSON文件路径')
    parser.add_argument('--output', '-o',
                       default='metrics_analysis',
                       help='输出文件前缀')
    parser.add_argument('--show-details', '-d',
                       action='store_true',
                       help='显示详细的案例信息')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.input).exists():
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return
    
    print(f"🔍 正在分析测试文件: {args.input}")
    
    # 提取数据
    try:
        extracted_data = extract_metrics_data(args.input)
        summary = create_metrics_summary(extracted_data)
        
        # 打印概览
        print_metrics_overview(summary)
        
        # 显示详细信息
        if args.show_details and extracted_data:
            print("\n" + "="*60)
            print("📋 详细案例信息 (前5个)")
            print("="*60)
            
            for i, case in enumerate(extracted_data[:5]):
                print(f"\n🔸 案例 {case['case_id']}: {case['case_name']}")
                print(f"   输入: {case['input'][:100]}...")
                print(f"   整体成功: {case['success']}")
                
                # 显示各个指标
                for key, value in case.items():
                    if key.endswith('_score'):
                        metric_name = key[:-6]
                        score = value
                        success = case.get(f'{metric_name}_success', False)
                        reason = case.get(f'{metric_name}_reason', '')[:100]
                        print(f"   📊 {metric_name}: 分数={score}, 成功={success}")
                        if reason:
                            print(f"      原因: {reason}...")
                        # 展示该指标的拆解维度分数
                        breakdown_items = [(bk, bv) for bk, bv in case.items() if bk.startswith(f'{metric_name}_breakdown_')]
                        if breakdown_items:
                            print(f"      拆解维度分数:")
                            for bk, bv in sorted(breakdown_items):
                                dim = bk.replace(f'{metric_name}_breakdown_', '')
                                print(f"        - {dim}: {bv}")
                # 展示预期工具调用（每个用例）
                if case.get('expected_tool_calls'):
                    preview = case['expected_tool_calls'][:200] + ("..." if len(case['expected_tool_calls']) > 200 else "")
                    print(f"   🔧 Expected tool calls: {preview}")
        
        # 保存结果
        save_to_formats(extracted_data, summary, args.output)
        
        print(f"\n✅ 分析完成! 共处理 {len(extracted_data)} 个测试案例")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()