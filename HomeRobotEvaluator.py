# -*- coding: utf-8 -*-
# @Author  : nehemiah.ouyang
# @Time    : 2024/9/27 20:56
# @Function:
import datetime
import logging
import re


class HomeRobotEvaluator:

    # log_analysis.py

    import re
    import logging

    # 配置日志记录
    logging.basicConfig(level=logging.INFO, filename='log_analysis.log', filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')

    def analyze_log(file_path):
        time_categories = {
            "非常快": [],
            "快": [],
            "慢": [],
            "很慢": []
        }

        # 正则表达式匹配日志中的执行时间
        time_pattern = re.compile(r"taking (\d+\.\d+) seconds")

        with open(file_path, 'r') as file:
            for line in file:
                match = time_pattern.search(line)
                if match:
                    duration = float(match.group(1))
                    # 根据执行时间分类
                    if duration < 20.0:
                        time_categories["非常快"].append(duration)
                    elif 20.0 <= duration < 30.0:
                        time_categories["快"].append(duration)
                    elif 30.0 <= duration < 50.0:
                        time_categories["慢"].append(duration)
                    else:
                        time_categories["很慢"].append(duration)

        # 打印分析结果
        for category, durations in time_categories.items():
            logging.info(f"{category}: {len(durations)} actions, durations: {durations}")


    
    def __init__(self, plans):
        self.plans = plans
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.improvements = 0
        self.exceptions_handled = 0
        self.response_times = []

    def evaluate_task_completion(self):
        """评估任务完成率"""
        total_tasks = len(self.plans)
        for act in self.plans:
            if act.get('status') == 'completed':
                self.completed_tasks += 1
        return self.completed_tasks / total_tasks if total_tasks > 0 else 0

    def count_failures(self):
        """统计失败次数"""
        for act in self.plans:
            if act.get('status') == 'failed':
                self.failed_tasks += 1
        return self.failed_tasks

    def assess_self_improvement(self):
        """评估自我完善程度"""
        # 假设每完成一次任务且未失败即视为一次改进
        return self.completed_tasks - self.failed_tasks

    def count_exception_handling(self):
        """统计异常处理次数"""
        for act in self.plans:
            if act.get('exception_handled'):
                self.exceptions_handled += 1
        return self.exceptions_handled

    def measure_response_speed(self):
        """统计反应速度"""
        for act in self.plans:
            start_time = datetime.datetime.strptime(act['start_time'], "%H:%M").time()
            end_time = datetime.datetime.strptime(act['end_time'], "%H:%M").time()
            if start_time < end_time:  # 同一天内
                delta = datetime.datetime.combine(datetime.date.today(), end_time) - datetime.datetime.combine(
                    datetime.date.today(), start_time)
                self.response_times.append(delta.total_seconds())
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0


def analyze_log(file_path=None):
    time_categories = {
        "非常快": [],
        "快": [],
        "慢": [],
        "很慢": []
    }

    # 正则表达式匹配日志中的执行时间
    # time_pattern = re.compile(r"taking (\d+\.\d+) seconds")
    # 更复杂的正则表达式
    time_pattern = re.compile(
        r"(?P<name>\w+) executed action '(?P<action>.+)' in '(?P<location>.+)' taking (?P<duration>\d+\.\d+) seconds")
    # match = re.search(r"(\w+) executed action '(.+?)' in .+? taking (\d+\.\d+) seconds\.", log_line)
    #
    # if match:
    #     person = match.group(1)
    #     action = match.group(2)
    #     time_taken = float(match.group(3))
    #
    #     print(f"人物名称: {person}")
    #     print(f"执行的动作: {action}")
    #     print(f"动作耗时: {time_taken} 秒")
    # else:
    #     print("没有找到匹配的信息")
    with open(file_path, 'r') as file:
        for line in file:
            # print(line)
            match = re.search(r"(\w+) executed action '(.+?)' in .+? taking (\d+\.\d+) seconds\.", line)
            match = time_pattern.search(line)
            if match:
                print(match.group(3))
                person = match.group(1)
                action = match.group(2)
                duration = float(match.group(4))
                print(f"人物名称: {person}")
                print(f"执行的动作: {action}")
                print(f"动作耗时: {match.group(4)} 秒")
                print()


                if duration < 20.0:
                    duration=str(duration)
                    time_categories["非常快"].append( person+action+"cost:"+duration)
                elif 20.0 <= duration < 30.0:
                    duration = str(duration)
                    time_categories["快"].append( person+action+"cost:"+duration)
                elif 30.0 <= duration < 50.0:
                    duration = str(duration)
                    time_categories["慢"].append( person+action+"cost:"+duration)
                else:
                    duration = str(duration)
                    time_categories["很慢"].append( person+action+"cost:"+duration)

    # 打印分析结果
    for category, durations in time_categories.items():
        print(f"{category}: {len(durations)} actions, durations: {durations}")
        logging.info(f"{category}: {len(durations)} actions, durations: {durations}")


def main():
    if __name__ == "__main__":
        analyze_log('simulation.log')
    # 假设这是从之前的代码中获取的计划列表
    # copied_combined_plans = [
    #     {"name": "Emma", "start_time": "08:00", "end_time": "08:30", "action": "Cooking", "status": "completed"},
    #     {"name": "Jason", "start_time": "08:30", "end_time": "09:00", "action": "Cleaning", "status": "failed"},
    #     {"name": "Tommie", "start_time": "09:00", "end_time": "09:30", "action": "Washing", "status": "completed",
    #      "exception_handled": True}
    # ]
    # 
    # evaluator = HomeRobotEvaluator(copied_combined_plans)
    # 
    # print(f"任务完成率: {evaluator.evaluate_task_completion() * 100}%")
    # print(f"失败次数: {evaluator.count_failures()}")
    # print(f"自我完善程度: {evaluator.assess_self_improvement()}")
    # print(f"异常处理次数: {evaluator.count_exception_handling()}")
    # print(f"平均反应速度: {evaluator.measure_response_speed()}秒")


if __name__ == "__main__":
    main()
