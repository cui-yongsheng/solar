# fileProcess.py
from datetime import datetime, timedelta
import pandas as pd
import os


class FileProcessor:
    """文件处理类，用于查找和读取指定格式的文件"""

    def __init__(self):
        """初始化文件处理器"""
        pass
    @staticmethod
    def find_file(folder_path, target_file):
        """
        递归查找指定文件夹及其子文件夹中是否包含目标文件
        Args:
            folder_path (str): 要搜索的根文件夹路径
            target_file (str): 目标文件名
        Returns:
            list: 找到的所有匹配文件的完整路径列表
        """
        found_files = []
        for root, dirs, files in os.walk(folder_path):
            if target_file in files:
                file_path = os.path.join(root, target_file)
                found_files.append(file_path)
        return found_files

    @staticmethod
    def parse_date(date_str):
        """
        解析日期字符串
        Args:
            date_str (str): 日期字符串，支持格式 "YYYYMMDD" 或 "YYYY-MM-DD"
        Returns:
            datetime: 解析后的日期对象
        """
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"无法解析日期格式: {date_str}")

    def find_files_by_date_range(self, root_path, start_date, end_date, target_file=None, file_pattern=None):
        """
        根据日期范围查找文件
        Args:
            root_path (str): 根目录路径
            start_date (str): 开始日期，格式为 "YYYYMMDD"
            end_date (str): 结束日期，格式为 "YYYYMMDD"
            target_file (str): 目标文件名（可选）
            file_pattern (str): 文件名模式，包含 {date} 占位符（可选）
        Returns:
            list: 找到的文件路径列表
        """
        found_files = []

        # 解析日期
        start = self.parse_date(start_date)
        end = self.parse_date(end_date)

        # 遍历日期范围内的每一天
        current = start
        while current <= end:
            # 支持两种格式构造目录路径：年/年月日 和 年/年-月-日
            year = current.strftime("%Y")
            date_formats = [
                current.strftime("%Y%m%d"),
                current.strftime("%Y-%m-%d")
            ]

            # 确定目标文件名
            current_target_file = target_file
            if file_pattern:
                date_str = current.strftime("%Y%m%d")
                current_target_file = file_pattern.format(date=date_str)

            file_found = False
            for date_str in date_formats:
                folder_path = os.path.join(root_path, year, date_str)
                if os.path.exists(folder_path):
                    files = self.find_file(folder_path, current_target_file)
                    if len(files) == 1:
                        found_files.append(files[0])
                        file_found = True
                        break
                    elif len(files) > 1:
                        print(f"{date_str} 目录下找到多个 {current_target_file} 文件,跳过")

            if not file_found:
                print(f"{current.strftime('%Y-%m-%d')} 对应的目录不存在或未找到文件")
            # 移动到下一天
            current += timedelta(days=1)
        return found_files

    def read_files(self, found_files, target_columns):
        """
        读取found_files中的所有CSV文件，提取指定字段并拼接数据
        Args:
            found_files (list): CSV文件路径列表
            target_columns (list): 需要提取的字段名列表
        Returns:
            pandas.DataFrame: 拼接后的数据
        """
        all_data = []

        # 检查是否有需要处理的文件
        if not found_files:
            print("未提供文件路径")
            return pd.DataFrame()

        total_files = len(found_files)
        print(f"开始处理 {total_files} 个文件...")

        for index, file_path in enumerate(found_files, 1):
            # 显示进度
            progress = (index / total_files) * 100
            print(f"进度: {index}/{total_files} ({progress:.1f}%) - 处理文件: {os.path.basename(file_path)}")

            try:
                # 读取CSV文件
                df = pd.read_csv(file_path, index_col=False, sep=',', encoding='gbk')
            except UnicodeDecodeError:
                print(f"文件{file_path}编码错误，尝试使用utf-8编码")
                try:
                    df = pd.read_csv(file_path, index_col=False, sep=',', encoding='utf-8')
                except Exception as e:
                    print(f"文件{file_path}读取失败，已跳过: {e}")
                    continue
            except FileNotFoundError:
                print(f"文件{file_path}不存在，已跳过")
                continue
            except Exception as e:
                print(f"文件{file_path}读取异常，已跳过: {e}")
                continue

            # 清理列名
            df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

            # 检查必需的时间列
            if '时间' not in df.columns:
                print(f"文件 {file_path} 缺少'时间'列，跳过该文件")
                continue

            # 获取存在的目标列
            existing_columns = [col for col in target_columns if col in df.columns]
            if not existing_columns:
                print(f"文件 {file_path} 中未找到任何目标字段，跳过该文件")
                continue

            # 确保时间列在结果中
            if '时间' not in existing_columns:
                existing_columns.insert(0, '时间')

            try:
                # 转换时间格式
                df['时间'] = pd.to_datetime(df['时间'], format='%Y-%m-%d %H:%M:%S')
                # 添加有效数据
                df = self._clean_data_units(df, existing_columns)
                all_data.append(df[existing_columns])
            except ValueError as e:
                print(f"文件 {file_path} 时间格式错误，跳过该文件: {e}")
                continue
            except Exception as e:
                print(f"文件 {file_path} 处理过程中发生错误，跳过该文件: {e}")
                continue

        # 拼接所有数据
        if all_data:
            try:
                concatenated_data = pd.concat(all_data, ignore_index=True)
                print(f"数据加载完成！共处理 {len(all_data)} 个有效文件，总数据行数: {len(concatenated_data)}")
                return concatenated_data
            except Exception as e:
                print(f"数据拼接失败: {e}")
                return pd.DataFrame()
        else:
            print("没有成功读取任何数据")
            return pd.DataFrame()

    @staticmethod
    def validate_and_clean_time_series(df, time_column='时间'):
        """
        验证时间序列数据的递增性并清理错误数据点（保持原始顺序）
        Args:
            df (pandas.DataFrame): 输入数据框
            time_column (str): 时间列名，默认为'时间'
        Returns:
            pandas.DataFrame: 清理后的时间序列数据
        """
        if df.empty:
            print("数据框为空，无需验证")
            return df

        if time_column not in df.columns:
            print(f"未找到时间列 {time_column}")
            return df

        df_copy = df.copy()

        # 检查是否有重复时间
        duplicates = df_copy[time_column].duplicated()
        if duplicates.any():
            duplicate_count = duplicates.sum()
            print(f"发现 {duplicate_count} 个重复时间点，将被移除")
            df_copy = df_copy[~duplicates].reset_index(drop=True)

        # 检查时间是否递增（不排序，只移除违反递增规则的点）
        if len(df_copy) > 1:
            time_series = df_copy[time_column]
            # 创建一个布尔掩码标识需要保留的行
            keep_mask = pd.Series([True] * len(df_copy))

            # 从第二行开始检查，如果当前时间小于等于前一个保留的时间，则标记为删除
            last_valid_time = time_series.iloc[0]
            for i in range(1, len(time_series)):
                if time_series.iloc[i] <= last_valid_time:
                    keep_mask.iloc[i] = False
                    print(f"移除时间倒退/重复数据点: {time_series.iloc[i]} (索引: {i})")
                else:
                    last_valid_time = time_series.iloc[i]

            # 应用掩码过滤数据
            df_filtered = df_copy[keep_mask].reset_index(drop=True)

            original_count = len(df_copy)
            filtered_count = len(df_filtered)
            removed_count = original_count - filtered_count

            if removed_count > 0:
                print(f"原始数据点: {original_count}, 清理后数据点: {filtered_count}, 移除数据点: {removed_count}")
            else:
                print("数据时间序列验证通过，无需清理")

            return df_filtered

        print("数据时间序列验证通过，无需清理")
        return df_copy

    @staticmethod
    def save_dataframe(df, output_path, file_format='csv', **kwargs):
        """
        高效保存DataFrame到文件
        Args:
            df (pandas.DataFrame): 要保存的数据框
            output_path (str): 输出文件路径
            file_format (str): 文件格式，支持 'csv', 'parquet'
            **kwargs: 传递给具体保存函数的额外参数
        Returns:
            bool: 保存成功返回True，否则返回False
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if file_format.lower() == 'csv':
                df.to_csv(output_path, index=False, encoding='utf-8-sig', **kwargs)
            elif file_format.lower() == 'parquet':
                # 最高效的列式存储格式
                df.to_parquet(output_path, index=False, **kwargs)
            else:
                raise ValueError(f"不支持的文件格式: {file_format}")

            print(f"数据已成功保存到 {output_path}")
            return True
        except Exception as e:
            print(f"保存文件失败: {e}")
            return False


    @staticmethod
    def _clean_data_units(df, target_columns):
        """
        清理数据中的单位信息
        Args:
            df (pandas.DataFrame): 数据框
            target_columns (list): 需要清理的列名列表
        Returns:
            pandas.DataFrame: 清理后的数据框
        """
        df_cleaned = df.copy()

        for col in target_columns:
            if col in df_cleaned.columns:
                # 检查列是否为时间类型
                if pd.api.types.is_datetime64_any_dtype(df_cleaned[col]):
                    continue  # 跳过时间列的处理
                # 如果是字符串类型，尝试提取数值部分
                if df_cleaned[col].dtype == 'object':
                    # 使用正则表达式提取数值
                    df_cleaned[col] = df_cleaned[col].astype(str).str.extract(r'([+-]?\d*\.?\d+)')[0]

                # 转换为数值类型
                try:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                except Exception as e:
                    print(f"列 {col} 单位清理失败: {e}")

        return df_cleaned