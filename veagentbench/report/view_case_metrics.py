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

#!/usr/bin/env python3
"""
简洁的测试案例指标查看器
快速查看每个test case的各项指标得分和原因
"""

import json
import pandas as pd
from pathlib import Path
import argparse

def view_case_metrics(test_run_file: str, case_id: int = None, show_reasons: bool = False):
    """
    查看测试案例的指标数据
    
    Args:
        test_run_file: 测试运行JSON文件路径
        case_id: 特定案例ID，None表示查看所有
        show_reasons: 是否显示详细原因
    """
    with open(test_run_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = data['testRunData']['testCases']
    
    if case_id is not None:
        if case_id >= len(test_cases):
            print(f"❌ 案例ID {case_id} 不存在，总共有 {len(test_cases)} 个案例")
            return
        test_cases = [test_cases[case_id]]
        start_id = case_id
    else:
        start_id = 0
    
    print("="*80)
    print("📊 测试案例指标详情")
    print("="*80)
    
    for i, test_case in enumerate(test_cases):
        current_id = start_id + i if case_id is not None else i
        
        print(f"\n🔸 案例 {current_id}: {test_case.get('name', f'test_case_{current_id}')}")
        print(f"📝 输入: {test_case.get('input', '')[:100]}...")
        print(f"✅ 整体成功: {test_case.get('success', False)}")
        print(f"⏱️  运行时间: {test_case.get('runDuration', 0):.2f}s")
        
        # 显示各个指标
        metrics_data = test_case.get('metricsData', [])
        
        if not metrics_data:
            print("   ⚠️  无指标数据")
            continue
        
        print("\n   📈 指标详情:")
        print("   " + "-"*60)
        
        for metric in metrics_data:
            name = metric.get('name', 'Unknown')
            score = metric.get('score', 0)
            success = metric.get('success', False)
            threshold = metric.get('threshold', 0)
            
            # 状态图标
            status_icon = "✅" if success else "❌"
            
            print(f"   {status_icon} {name}")
            print(f"      分数: {score:.3f} (阈值: {threshold})")
            
            if show_reasons:
                reason = metric.get('reason', '')
                if reason:
                    # 截断过长的原因
                    if len(reason) > 150:
                        reason = reason[:150] + "..."
                    print(f"      原因: {reason}")
            
            print()

def create_metrics_table(test_run_file: str, output_file: str = None):
    """
    创建指标对比表格
    
    Args:
        test_run_file: 测试运行JSON文件路径
        output_file: 输出文件路径
    """
    with open(test_run_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = data['testRunData']['testCases']
    
    # 准备表格数据
    table_data = []
    
    for i, test_case in enumerate(test_cases):
        row = {
            'Case ID': i,
            'Case Name': test_case.get('name', f'test_case_{i}'),
            'Overall Success': '✅' if test_case.get('success', False) else '❌',
            'Duration (s)': f"{test_case.get('runDuration', 0):.2f}"
        }
        
        # 添加各个指标的分数
        metrics_data = test_case.get('metricsData', [])
        for metric in metrics_data:
            name = metric.get('name', 'Unknown')
            score = metric.get('score', 0)
            success = metric.get('success', False)
            
            # 使用简化的列名
            col_name = name.replace(' ', '_').replace('-', '_')
            row[f'{col_name}_Score'] = f"{score:.3f}"
            row[f'{col_name}_Pass'] = '✅' if success else '❌'
        
        table_data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(table_data)
    
    # 显示表格
    print("\n" + "="*120)
    print("📊 指标对比表格")
    print("="*120)
    
    # 设置显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 15)
    
    print(df.to_string(index=False))
    
    # 保存到文件
    if output_file:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✅ 表格已保存到: {output_file}")

def show_metrics_statistics(test_run_file: str):
    """
    显示指标统计信息
    
    Args:
        test_run_file: 测试运行JSON文件路径
    """
    with open(test_run_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = data['testRunData']['testCases']
    
    # 收集所有指标数据
    metrics_stats = {}
    
    for test_case in test_cases:
        metrics_data = test_case.get('metricsData', [])
        
        for metric in metrics_data:
            name = metric.get('name', 'Unknown')
            score = metric.get('score', 0)
            success = metric.get('success', False)
            
            if name not in metrics_stats:
                metrics_stats[name] = {
                    'scores': [],
                    'successes': [],
                    'total': 0
                }
            
            metrics_stats[name]['scores'].append(score)
            metrics_stats[name]['successes'].append(success)
            metrics_stats[name]['total'] += 1
    
    # 显示统计信息
    print("\n" + "="*80)
    print("📈 指标统计汇总")
    print("="*80)
    
    for name, stats in metrics_stats.items():
        scores = stats['scores']
        successes = stats['successes']
        
        avg_score = sum(scores) / len(scores) if scores else 0
        success_rate = sum(successes) / len(successes) if successes else 0
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        print(f"\n🔹 {name}")
        print(f"   平均分数: {avg_score:.3f}")
        print(f"   分数范围: {min_score:.3f} - {max_score:.3f}")
        print(f"   通过率: {success_rate:.1%}")
        print(f"   评估次数: {stats['total']}")

def main():
    parser = argparse.ArgumentParser(description='查看测试案例的指标数据')
    parser.add_argument('--input', '-i', 
                       default='.deepeval/.latest_test_run.json',
                       help='输入的测试运行JSON文件路径')
    parser.add_argument('--case', '-c', type=int,
                       help='查看特定案例ID')
    parser.add_argument('--reasons', '-r', action='store_true',
                       help='显示详细原因')
    parser.add_argument('--table', '-t', action='store_true',
                       help='显示对比表格')
    parser.add_argument('--stats', '-s', action='store_true',
                       help='显示统计信息')
    parser.add_argument('--output', '-o',
                       help='保存表格到文件')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.input).exists():
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return
    
    try:
        if args.table:
            create_metrics_table(args.input, args.output)
        elif args.stats:
            show_metrics_statistics(args.input)
        else:
            view_case_metrics(args.input, args.case, args.reasons)
            
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()