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
import ast



def safe_parse_json_string(json_str: str) -> Any:
    """
    安全地解析可能包含Python字面量的JSON字符串
    """
    try:
        # 首先尝试标准JSON解析
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # 替换Python字面量
            json_str = json_str.replace("true", "True").replace("false", "False").replace("null", "None")
            # 使用ast.literal_eval
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            # 最后尝试eval
            return eval(json_str)

def extract_metrics_data(test_run_file: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    从测试运行文件中提取每个test case的metrics数据
    
    Args:
        test_run_file: 测试运行JSON文件路径
        
    Returns:
        包含所有task和dataset的metrics数据的字典
        格式: {task_name: {dataset_name: [test_cases...]}}
    """
    
    with open(test_run_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    task_testcases_all = _extract_metrics_data(data)
    return task_testcases_all

def _extract_metrics_data(data):
    task_testcases_all = {}

    # 支持两种格式：直接数组或包含testRunData的对象
    if isinstance(data, list):
        # 直接是任务结果数组
        for task_result in data:
            task_name = task_result['task_name']
            task_testcases_all[task_name] = {}
            
            if 'result' in task_result and isinstance(task_result['result'], dict):
                measure_result = task_result['result'].get('result', '')
                if measure_result and isinstance(measure_result, str):
                    try:
                        # 解析measure_result字符串
                        measure_data = safe_parse_json_string(measure_result)
                        
                        if isinstance(measure_data, list) and len(measure_data) > 0:
                            # measure_data是数组，处理每个dataset的数据
                            for dataset_item in measure_data:
                                dataset_name = dataset_item.get("dataset_name", "default")
                                task_testcases_all[task_name].setdefault(
                                    dataset_name, [])
                                
                                if isinstance(dataset_item, dict) and 'measure_result' in dataset_item:
                                    test_results = dataset_item.get('measure_result', '')
                                    if test_results and isinstance(test_results, str):
                                        test_data = safe_parse_json_string(test_results)
                                        
                                        if 'test_results' in test_data:
                                            extracted_cases = extract_metrics_data_from_testcases(test_data['test_results'])
                                            # 为每个case添加dataset_name信息
                                            for case in extracted_cases:
                                                case['dataset_name'] = dataset_name
                                                case['task_name'] = task_name
                                            task_testcases_all[task_name][dataset_name].extend(extracted_cases)
                                
                    except Exception as e:
                        print(f"⚠️ 解析任务结果失败: {e}")
                        continue
    
    return task_testcases_all


def extract_metrics_data_from_testcases(test_cases: List[Dict[str, Any]]):
    
    extracted_data = []
    
    for i, test_case in enumerate(test_cases):
        case_data = {
            'case_id': i,
            'case_name': test_case.get('name', f'test_case_{i}'),
            'input': test_case.get('input', ''),
            'actual_output': test_case.get('actual_output', ''),
            'expected_output': test_case.get('expected_output', ''),
            'success': test_case.get('success', False),
            'run_duration': test_case.get('runDuration', 0)
        }
        
        # 提取metrics_data中的每个指标（支持多种键名格式）
        metrics_data = test_case.get('metrics_data') or test_case.get('metricsData', [])
        
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
        
        # 预期工具调用（expected_tool_calls），兼容多种键名格式
        expected_calls = test_case.get('expected_tool_calls') or test_case.get('expectedToolCalls')
        if expected_calls is None:
            execution_data = test_case.get('execution_data') or test_case.get('executionData') or {}
            expected_calls = execution_data.get('expected_tool_calls') or execution_data.get('expectedToolCalls')
        # 以JSON字符串形式保存到详细表，便于查看
        try:
            case_data['expected_tool_calls'] = json.dumps(expected_calls, ensure_ascii=False) if expected_calls is not None else ''
        except Exception:
            case_data['expected_tool_calls'] = str(expected_calls) if expected_calls is not None else ''
        
        extracted_data.append(case_data)
    
    return extracted_data

def create_metrics_summary(extracted_data: List[Dict[str, Any]], group_by_dataset: bool = False) -> Dict[str, Any]:
    """
    创建metrics汇总统计，包含breakdown维度指标的汇总
    
    Args:
        extracted_data: 提取的metrics数据
        group_by_dataset: 是否按dataset分组统计
        
    Returns:
        汇总统计信息
    """
    if not extracted_data:
        return {}
    
    # 找出所有的指标名称和对应的breakdown维度
    metric_names = set()
    breakdown_dimensions = {}  # metric_name -> set of breakdown dimensions
    
    for case in extracted_data:
        for key in case.keys():
            if key.endswith('_score'):
                metric_name = key[:-6]  # 移除'_score'后缀
                metric_names.add(metric_name)
                
                # 收集该指标的breakdown维度
                breakdown_prefix = f'{metric_name}_breakdown_'
                for case_key in case.keys():
                    if case_key.startswith(breakdown_prefix) and case.get(case_key) is not None:
                        dim_name = case_key[len(breakdown_prefix):]
                        if metric_name not in breakdown_dimensions:
                            breakdown_dimensions[metric_name] = set()
                        breakdown_dimensions[metric_name].add(dim_name)
    
    def calculate_breakdown_stats(cases, metric_name):
        """计算指定指标的breakdown维度统计"""
        breakdown_stats = {}
        if metric_name in breakdown_dimensions:
            for dim_name in breakdown_dimensions[metric_name]:
                dim_key = f'{metric_name}_breakdown_{dim_name}'
                dim_values = [case[dim_key] for case in cases if case.get(dim_key) is not None]
                if dim_values:
                    breakdown_stats[dim_name] = {
                        'avg_score': sum(dim_values) / len(dim_values),
                        'min_score': min(dim_values),
                        'max_score': max(dim_values),
                        'total_evaluations': len(dim_values)
                    }
        return breakdown_stats
    
    if group_by_dataset and extracted_data and 'dataset_name' in extracted_data[0]:
        # 按dataset分组统计
        datasets = {}
        for case in extracted_data:
            dataset_name = case['dataset_name']
            if dataset_name not in datasets:
                datasets[dataset_name] = []
            datasets[dataset_name].append(case)
        
        summary = {
            'total_cases': len(extracted_data),
            'overall_success_rate': sum(1 for case in extracted_data if case['success']) / len(extracted_data),
            'datasets_summary': {},
            'metrics_summary': {}
        }
        
        # 为每个dataset创建统计
        for dataset_name, dataset_cases in datasets.items():
            dataset_summary = {
                'total_cases': len(dataset_cases),
                'overall_success_rate': sum(1 for case in dataset_cases if case['success']) / len(dataset_cases),
                'metrics_summary': {}
            }
            
            # 为每个指标计算统计信息
            for metric_name in metric_names:
                score_key = f'{metric_name}_score'
                success_key = f'{metric_name}_success'
                
                scores = [case[score_key] for case in dataset_cases if case.get(score_key) is not None]
                successes = [case[success_key] for case in dataset_cases if success_key in case]
                
                if scores:
                    metric_summary = {
                        'avg_score': sum(scores) / len(scores),
                        'min_score': min(scores),
                        'max_score': max(scores),
                        'success_rate': sum(successes) / len(successes) if successes else 0,
                        'total_evaluations': len(scores)
                    }
                    
                    # 添加breakdown维度统计
                    breakdown_stats = calculate_breakdown_stats(dataset_cases, metric_name)
                    if breakdown_stats:
                        metric_summary['breakdown_summary'] = breakdown_stats
                    
                    dataset_summary['metrics_summary'][metric_name] = metric_summary
            
            summary['datasets_summary'][dataset_name] = dataset_summary
        
        # 计算整体的metrics_summary
        for metric_name in metric_names:
            score_key = f'{metric_name}_score'
            success_key = f'{metric_name}_success'
            
            scores = [case[score_key] for case in extracted_data if case.get(score_key) is not None and case[score_key] is not -1]
            successes = [case[success_key] for case in extracted_data if success_key in case]
            
            if scores:
                metric_summary = {
                    'avg_score': sum(scores) / len(scores),
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'success_rate': sum(successes) / len(successes) if successes else 0,
                    'total_evaluations': len(scores)
                }
                
                # 添加整体的breakdown维度统计
                breakdown_stats = calculate_breakdown_stats(extracted_data, metric_name)
                if breakdown_stats:
                    metric_summary['breakdown_summary'] = breakdown_stats
                
                summary['metrics_summary'][metric_name] = metric_summary
        
        return summary
    else:
        # 原始的单dataset统计逻辑
        summary = {
            'total_cases': len(extracted_data),
            'overall_success_rate': sum(1 for case in extracted_data if case['success']) / len(extracted_data),
            'metrics_summary': {}
        }
        
        # 为每个指标计算统计信息
        for metric_name in metric_names:
            score_key = f'{metric_name}_score'
            success_key = f'{metric_name}_success'
            
            scores = [case[score_key] for case in extracted_data if case.get(score_key) is not None and case[score_key] is not -1]
            successes = [case[success_key] for case in extracted_data if success_key in case]
            
            if scores:
                metric_summary = {
                    'avg_score': sum(scores) / len(scores),
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'success_rate': sum(successes) / len(successes) if successes else 0,
                    'total_evaluations': len(scores)
                }
                
                # 添加breakdown维度统计
                breakdown_stats = calculate_breakdown_stats(extracted_data, metric_name)
                if breakdown_stats:
                    metric_summary['breakdown_summary'] = breakdown_stats
                
                summary['metrics_summary'][metric_name] = metric_summary
        
        return summary

def save_to_formats(extracted_data: List[Dict[str, Any]], summary: Dict[str, Any], output_prefix: str, multi_dataset: bool = False):
    """
    保存数据到多种格式
    
    Args:
        extracted_data: 提取的数据
        summary: 汇总统计
        output_prefix: 输出文件前缀
        multi_dataset: 是否为多dataset模式
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
        
        # 添加dataset名称（如果存在）
        if 'dataset_name' in case:
            row['dataset_name'] = case['dataset_name']
        
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
.dataset-section { background: #f9f9f9; padding: 16px; border-radius: 8px; margin: 16px 0; }
</style>
</head><body>
""")
        html_parts.append("<h1>Metrics Report / 指标评估报告</h1>")
        html_parts.append(f"<div class='kpi'><div class='card'><b>Total cases</b><br>{summary.get('total_cases', 0)}</div>")
        overall_rate = summary.get('overall_success_rate', 0)
        html_parts.append(f"<div class='card'><b>Overall success rate</b><br>{overall_rate:.2%}</div></div>")
        
        # 多dataset模式：分别显示每个dataset的统计
        if multi_dataset and 'datasets_summary' in summary:
            html_parts.append("<div class='section'><h2>Datasets Summary / 数据集汇总</h2>")
            
            # 汇总表（所有dataset的概览）
            datasets_overview = []
            for dataset_name, dataset_stats in summary['datasets_summary'].items():
                overview_row = {
                    'dataset_name': dataset_name,
                    'total_cases': dataset_stats['total_cases'],
                    'overall_success_rate': f"{dataset_stats['overall_success_rate']:.2%}",
                    'metrics_count': len(dataset_stats.get('metrics_summary', {}))
                }
                datasets_overview.append(overview_row)
            
            if datasets_overview:
                overview_df = pd.DataFrame(datasets_overview)
                html_parts.append("<h3>Dataset Overview / 数据集概览</h3>")
                html_parts.append(overview_df.to_html(index=False))
            
            # 每个dataset的详细统计
            for dataset_name, dataset_stats in summary['datasets_summary'].items():
                html_parts.append(f"<div class='dataset-section'>")
                html_parts.append(f"<h3>Dataset: {dataset_name}</h3>")
                html_parts.append(f"<div class='kpi'><div class='card'><b>Cases</b><br>{dataset_stats['total_cases']}</div>")
                html_parts.append(f"<div class='card'><b>Success Rate</b><br>{dataset_stats['overall_success_rate']:.2%}</div></div>")
                
                # dataset的metrics统计
                dataset_metrics = []
                for metric_name, metric_stats in dataset_stats.get('metrics_summary', {}).items():
                    metric_row = {'metric': metric_name}
                    metric_row.update(metric_stats)
                    dataset_metrics.append(metric_row)
                
                if dataset_metrics:
                    dataset_metrics_df = pd.DataFrame(dataset_metrics)
                    html_parts.append("<h4>Metrics Summary / 指标汇总</h4>")
                    html_parts.append(dataset_metrics_df.to_html(index=False))
                
                html_parts.append("</div>")
        
        # 整体汇总表（metrics_summary）
        html_parts.append("<div class='section'><h2>Overall Metrics Summary / 整体指标汇总</h2>")
        metrics_summary = summary.get('metrics_summary', {})
        summary_rows = []
        for name, stats in metrics_summary.items():
            row = {'metric': name}
            row.update(stats)
            summary_rows.append(row)
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            html_parts.append(summary_df.to_html(index=False))
        else:
            html_parts.append("<div class='small'>No metrics summary.</div>")
        html_parts.append("</div>")
        
        # Breakdown维度汇总表格
        if multi_dataset and 'datasets_summary' in summary:
            html_parts.append("<div class='section'><h2>Breakdown Dimensions Summary by Dataset / 各数据集拆解维度汇总</h2>")
            
            # 收集所有breakdown维度的数据
            breakdown_data = []
            for dataset_name, dataset_stats in summary['datasets_summary'].items():
                for metric_name, metric_stats in dataset_stats.get('metrics_summary', {}).items():
                    if 'breakdown_summary' in metric_stats:
                        for dim_name, dim_stats in metric_stats['breakdown_summary'].items():
                            breakdown_data.append({
                                'dataset_name': dataset_name,
                                'metric_name': metric_name,
                                'dimension_name': dim_name,
                                'avg_score': dim_stats['avg_score'],
                                'min_score': dim_stats['min_score'],
                                'max_score': dim_stats['max_score'],
                                'total_evaluations': dim_stats['total_evaluations']
                            })
            
            if breakdown_data:
                breakdown_df = pd.DataFrame(breakdown_data)
                
                # 创建透视表，按数据集和维度展示
                pivot_table = breakdown_df.pivot_table(
                    index=['metric_name', 'dimension_name'],
                    columns='dataset_name',
                    values='avg_score',
                    aggfunc='mean'
                ).round(3)
                
                html_parts.append("<h3>Average Scores by Dataset and Dimension / 各数据集维度平均分</h3>")
                html_parts.append(pivot_table.to_html())
                
                # 显示详细数据表
                html_parts.append("<h3>Detailed Breakdown Data / 详细拆解数据</h3>")
                html_parts.append(breakdown_df.to_html(index=False))
            else:
                html_parts.append("<div class='small'>No breakdown dimensions found.</div>")
            
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

def print_metrics_overview(summary: Dict[str, Any], multi_dataset: bool = False):
    """
    打印metrics概览，包含breakdown维度统计
    
    Args:
        summary: 汇总统计信息
        multi_dataset: 是否为多dataset模式
    """
    print("\n" + "="*60)
    print("📊 METRICS 数据概览")
    print("="*60)
    
    print(f"📋 总测试案例数: {summary['total_cases']}")
    print(f"✅ 整体成功率: {summary['overall_success_rate']:.2%}")
    
    # 多dataset模式：显示每个dataset的概览
    if multi_dataset and 'datasets_summary' in summary:
        print("\n📊 各数据集统计:")
        print("-" * 60)
        for dataset_name, dataset_stats in summary['datasets_summary'].items():
            print(f"\n📁 Dataset: {dataset_name}")
            print(f"   案例数: {dataset_stats['total_cases']}")
            print(f"   成功率: {dataset_stats['overall_success_rate']:.2%}")
    
    print("\n📈 各指标统计:")
    print("-" * 60)
    
    for metric_name, stats in summary['metrics_summary'].items():
        print(f"\n🔹 {metric_name.replace('_', ' ').title()}")
        print(f"   平均分数: {stats['avg_score']:.3f}")
        print(f"   分数范围: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
        print(f"   成功率: {stats['success_rate']:.2%}")
        print(f"   评估次数: {stats['total_evaluations']}")
        
        # 显示breakdown维度统计
        if 'breakdown_summary' in stats:
            print(f"   📊 拆解维度统计:")
            for dim_name, dim_stats in stats['breakdown_summary'].items():
                print(f"     - {dim_name.replace('_', ' ').title()}:")
                print(f"       平均分数: {dim_stats['avg_score']:.3f}")
                print(f"       分数范围: {dim_stats['min_score']:.3f} - {dim_stats['max_score']:.3f}")
                print(f"       评估次数: {dim_stats['total_evaluations']}")

def process_single_task(task_name: str, task_data: Dict[str, List[Dict[str, Any]]], output_prefix: str, show_details: bool = False):
    """
    处理单个task的数据，生成独立的报告文件
    
    Args:
        task_name: 任务名称
        task_data: 该任务的数据，格式为 {dataset_name: [test_cases...]}
        output_prefix: 输出文件前缀
        show_details: 是否显示详细信息
    """
    print(f"\n📊 处理任务: {task_name}")
    
    # 合并所有dataset的测试案例
    all_task_cases = []
    total_cases = 0
    
    for dataset_name, cases in task_data.items():
        all_task_cases.extend(cases)
        total_cases += len(cases)
        print(f"   Dataset '{dataset_name}': {len(cases)} 个测试案例")
    
    if not all_task_cases:
        print(f"   ⚠️ 任务 {task_name} 没有测试案例")
        return
    
    print(f"   总计: {total_cases} 个测试案例")
    
    # 判断是否使用多dataset模式（单个任务内有多个dataset）
    multi_dataset = len(task_data) > 1
    
    # 创建汇总统计
    summary = create_metrics_summary(all_task_cases, group_by_dataset=multi_dataset)
    
    # 打印概览
    print_metrics_overview(summary, multi_dataset=multi_dataset)
    
    # 显示详细信息
    if show_details and all_task_cases:
        print("\n" + "="*60)
        print("📋 详细案例信息 (前5个)")
        print("="*60)
        
        for i, case in enumerate(all_task_cases[:5]):
            dataset_info = f" [{case['dataset_name']}]" if 'dataset_name' in case else ""
            print(f"\n🔸 案例 {case['case_id']}{dataset_info}: {case['case_name']}")
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
    
    # 为每个任务生成独立的输出文件名
    task_output_prefix = f"{output_prefix}/{task_name}"
    
    # 保存结果
    save_to_formats(all_task_cases, summary, task_output_prefix, multi_dataset=multi_dataset)
    
    print(f"\n✅ 任务 {task_name} 分析完成! 共处理 {total_cases} 个测试案例")

def main():
    parser = argparse.ArgumentParser(description='提取测试案例的metrics数据')
    parser.add_argument('--input', '-i', 
                       nargs='+',  # 支持多个输入文件
                       default=['.deepeval/.latest_test_run.json'],
                       help='输入的测试运行JSON文件路径（可指定多个）')
    parser.add_argument('--output', '-o',
                       default='metrics_analysis',
                       help='输出文件前缀')
    parser.add_argument('--show-details', '-d',
                       action='store_true',
                       help='显示详细的案例信息')
    parser.add_argument('--dataset-names', '-n',
                       nargs='+',
                       help='为每个输入文件指定dataset名称（可选）')
    parser.add_argument('--per-task', '-t',
                       action='store_true',
                       help='为每个task生成独立的报告文件')
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_files = args.input
    dataset_names = args.dataset_names or []
    
    # 确保dataset_names数量与输入文件数量匹配
    if dataset_names and len(dataset_names) != len(input_files):
        print(f"❌ 错误: dataset_names数量({len(dataset_names)})与输入文件数量({len(input_files)})不匹配")
        return
    
    # 检查所有输入文件是否存在
    for input_file in input_files:
        if not Path(input_file).exists():
            print(f"❌ 错误: 输入文件不存在: {input_file}")
            return
    
    print(f"🔍 正在分析 {len(input_files)} 个测试文件")
    for i, input_file in enumerate(input_files):
        dataset_name = dataset_names[i] if i < len(dataset_names) else None
        print(f"  - {input_file}" + (f" (dataset: {dataset_name})" if dataset_name else ""))
    
    # 提取所有数据
    try:
        all_extracted_data = []
        
        # 处理每个输入文件
        for i, input_file in enumerate(input_files):
            print(f"\n📊 处理文件 {i+1}/{len(input_files)}: {input_file}")
            
            # 提取该文件的数据
            task_data_dict = extract_metrics_data(input_file)
            
            if args.per_task:
                # 为每个task生成独立报告
                for task_name, task_data in task_data_dict.items():
                    process_single_task(task_name, task_data, args.output, args.show_details)
            else:
                # 传统模式：合并所有数据
                for task_name, task_data in task_data_dict.items():
                    for dataset_name, cases in task_data.items():
                        all_extracted_data.extend(cases)
                
                print(f"   提取了 {len(all_extracted_data)} 个测试案例")
        
        if not args.per_task:
            # 传统模式：生成总体报告
            if all_extracted_data:
                # 判断是否使用多dataset模式
                multi_dataset = any('dataset_name' in case for case in all_extracted_data)
                
                # 创建汇总统计
                summary = create_metrics_summary(all_extracted_data, group_by_dataset=multi_dataset)
                
                # 打印概览
                print_metrics_overview(summary, multi_dataset=multi_dataset)
                
                # 保存结果
                save_to_formats(all_extracted_data, summary, args.output, multi_dataset=multi_dataset)
                
                print(f"\n✅ 分析完成! 共处理 {len(all_extracted_data)} 个测试案例")
            else:
                print("⚠️ 没有提取到任何测试案例")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
