import sys
import re
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QTextEdit,
                             QTabWidget, QComboBox, QGroupBox, QSpinBox)
from PyQt5.QtCore import Qt, QTimer


class MemoryBlock:
    """内存块类，用于动态分区分配"""

    def __init__(self, start, size, is_free=True, job_id=None):
        self.start = start  # 起始地址
        self.size = size  # 大小
        self.is_free = is_free  # 是否空闲
        self.job_id = job_id  # 占用该分区的作业ID


class DynamicPartitionManager:
    """动态分区分配管理器"""

    def __init__(self, total_memory_size=640):
        # 初始化一个空闲内存块
        self.memory_blocks = [MemoryBlock(0, total_memory_size)]
        self.total_memory_size = total_memory_size

    def allocate_first_fit(self, job_id, size):
        """首次适应算法"""
        for i, block in enumerate(self.memory_blocks):
            if block.is_free and block.size >= size:
                # 找到足够大的空闲分区
                if block.size == size:
                    # 如果分区大小正好等于请求大小，直接分配
                    block.is_free = False
                    block.job_id = job_id
                else:
                    # 分割空闲分区
                    self.memory_blocks.insert(i + 1, MemoryBlock(block.start + size, block.size - size))
                    block.size = size
                    block.is_free = False
                    block.job_id = job_id
                return True
        return False  # 没有找到合适的空闲分区

    def allocate_best_fit(self, job_id, size):
        """最佳适应算法"""
        best_fit_index = -1
        min_fragment = float('inf')

        for i, block in enumerate(self.memory_blocks):
            if block.is_free and block.size >= size:
                fragment = block.size - size
                if fragment < min_fragment:
                    min_fragment = fragment
                    best_fit_index = i

        if best_fit_index != -1:
            block = self.memory_blocks[best_fit_index]
            if block.size == size:
                # 如果分区大小正好等于请求大小，直接分配
                block.is_free = False
                block.job_id = job_id
            else:
                # 分割空闲分区
                self.memory_blocks.insert(best_fit_index + 1, MemoryBlock(block.start + size, block.size - size))
                block.size = size
                block.is_free = False
                block.job_id = job_id
            return True
        return False  # 没有找到合适的空闲分区

    def free_memory(self, job_id):
        """释放内存"""
        found = False
        for i, block in enumerate(self.memory_blocks):
            if not block.is_free and block.job_id == job_id:
                block.is_free = True
                block.job_id = None
                found = True
                # 合并相邻的空闲分区
                self._merge_adjacent_free_blocks()
                break
        return found

    def _merge_adjacent_free_blocks(self):
        """合并相邻的空闲分区"""
        i = 0
        while i < len(self.memory_blocks) - 1:
            if self.memory_blocks[i].is_free and self.memory_blocks[i + 1].is_free:
                # 合并两个相邻的空闲分区
                self.memory_blocks[i].size += self.memory_blocks[i + 1].size
                self.memory_blocks.pop(i + 1)
            else:
                i += 1

    def get_memory_status(self):
        """获取内存状态的字符串表示"""
        result = []
        for block in self.memory_blocks:
            status = "空闲" if block.is_free else f"作业{block.job_id}"
            result.append(f"起始地址: {block.start}K, 大小: {block.size}K, 状态: {status}")
        return result


class Page:
    """页面类"""

    def __init__(self, page_number):
        self.page_number = page_number  # 页号
        self.in_memory = False  # 是否在内存中
        self.frame_number = None  # 如果在内存中，对应的物理帧号
        self.load_time = None  # 加载到内存的时间(FIFO算法使用)
        self.last_access_time = None  # 最后访问时间(LRU算法使用)


class PageReplacementManager:
    """页面置换管理器"""

    def __init__(self, page_count=32, frame_count=4, instructions_per_page=10):
        self.page_count = page_count  # 总页数
        self.frame_count = frame_count  # 内存帧数
        self.instructions_per_page = instructions_per_page  # 每页指令数
        self.pages = [Page(i) for i in range(page_count)]  # 页表
        self.frames = [None] * frame_count  # 内存帧
        self.clock = 0  # 时钟，用于记录时间
        self.page_faults = 0  # 缺页次数
        self.accessed_instructions = 0  # 已访问的指令数

    def reset(self):
        """重置状态"""
        self.pages = [Page(i) for i in range(self.page_count)]
        self.frames = [None] * self.frame_count
        self.clock = 0
        self.page_faults = 0
        self.accessed_instructions = 0

    def access_instruction(self, instruction_number):
        """访问指令"""
        self.accessed_instructions += 1
        page_number = instruction_number // self.instructions_per_page
        offset = instruction_number % self.instructions_per_page

        page = self.pages[page_number]
        page.last_access_time = self.clock  # 更新最后访问时间

        result = {"instruction": instruction_number, "page": page_number, "offset": offset}

        if page.in_memory:
            # 页面在内存中
            physical_address = page.frame_number * self.instructions_per_page + offset
            result["status"] = "命中"
            result["physical_address"] = physical_address
        else:
            # 页面不在内存中，发生缺页
            self.page_faults += 1
            result["status"] = "缺页"

            # 查找空闲帧
            free_frame = self._find_free_frame()
            if free_frame is not None:
                # 有空闲帧
                page.in_memory = True
                page.frame_number = free_frame
                page.load_time = self.clock
                self.frames[free_frame] = page_number
                physical_address = free_frame * self.instructions_per_page + offset
                result["physical_address"] = physical_address
                result["replaced"] = False
            else:
                # 没有空闲帧，需要置换
                replaced_frame, replaced_page = self._replace_page()
                self.pages[replaced_page].in_memory = False
                self.pages[replaced_page].frame_number = None

                page.in_memory = True
                page.frame_number = replaced_frame
                page.load_time = self.clock
                self.frames[replaced_frame] = page_number
                physical_address = replaced_frame * self.instructions_per_page + offset
                result["physical_address"] = physical_address
                result["replaced"] = True
                result["replaced_page"] = replaced_page

        self.clock += 1
        return result

    def _find_free_frame(self):
        """查找空闲帧"""
        for i, frame in enumerate(self.frames):
            if frame is None:
                return i
        return None

    def _replace_page_fifo(self):
        """FIFO页面置换算法"""
        # 寻找最早加载的页面
        min_load_time = float('inf')
        min_frame = -1
        for i, page_number in enumerate(self.frames):
            if self.pages[page_number].load_time < min_load_time:
                min_load_time = self.pages[page_number].load_time
                min_frame = i

        return min_frame, self.frames[min_frame]

    def _replace_page_lru(self):
        """LRU页面置换算法"""
        # 寻找最久未使用的页面
        min_access_time = float('inf')
        min_frame = -1
        for i, page_number in enumerate(self.frames):
            if self.pages[page_number].last_access_time < min_access_time:
                min_access_time = self.pages[page_number].last_access_time
                min_frame = i

        return min_frame, self.frames[min_frame]

    def _replace_page(self):
        """根据当前设置的算法进行页面置换"""
        # 这个方法将在UI中被重写，根据选择的算法来决定使用哪种置换方法
        pass

    def get_page_fault_rate(self):
        """获取缺页率"""
        if self.accessed_instructions == 0:
            return 0
        return self.page_faults / self.accessed_instructions


class InstructionGenerator:
    """指令生成器"""

    def __init__(self, instruction_count=320):
        self.instruction_count = instruction_count
        self.instructions = []  # 存储生成的指令序列

    def generate_instructions(self):
        """生成指令序列"""
        self.instructions = []

        # 随机选取起始执行指令
        current_pos = random.randint(0, self.instruction_count - 1)
        self.instructions.append(current_pos)

        while len(self.instructions) < self.instruction_count:
            # 顺序执行下一条指令
            next_pos = (current_pos + 1) % self.instruction_count
            self.instructions.append(next_pos)
            current_pos = next_pos

            if len(self.instructions) >= self.instruction_count:
                break

            # 跳转到前地址部分
            if current_pos > 0:
                prev_pos = random.randint(0, current_pos - 1)
                self.instructions.append(prev_pos)
                current_pos = prev_pos

            if len(self.instructions) >= self.instruction_count:
                break

            # 顺序执行下一条指令
            next_pos = (current_pos + 1) % self.instruction_count
            self.instructions.append(next_pos)
            current_pos = next_pos

            if len(self.instructions) >= self.instruction_count:
                break

            # 跳转到后地址部分
            if current_pos < self.instruction_count - 1:
                next_pos = random.randint(current_pos + 1, self.instruction_count - 1)
                self.instructions.append(next_pos)
                current_pos = next_pos

        return self.instructions[:self.instruction_count]


class MemorySimulationApp(QMainWindow):
    """内存管理模拟应用"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("内存管理模拟系统")
        self.setGeometry(100, 100, 1000, 700)

        # 创建模拟器实例
        self.dynamic_manager = DynamicPartitionManager(640)  # 640K内存
        self.page_manager = PageReplacementManager(32, 4, 10)  # 32页，4个内存帧，每页10条指令
        self.instruction_generator = InstructionGenerator(320)  # 320条指令

        # 设置置换算法
        self.current_algorithm = "FIFO"  # 默认使用FIFO
        self.page_manager._replace_page = self.page_manager._replace_page_fifo

        # 初始化UI
        self._init_ui()

    def _init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # 创建选项卡
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)

        # 动态分区分配选项卡
        dynamic_tab = QWidget()
        tab_widget.addTab(dynamic_tab, "动态分区分配")

        # 页面置换选项卡
        page_tab = QWidget()
        tab_widget.addTab(page_tab, "页面置换")

        # 设置动态分区分配选项卡
        self._setup_dynamic_tab(dynamic_tab)

        # 设置页面置换选项卡
        self._setup_page_tab(page_tab)

    def _setup_dynamic_tab(self, tab):
        """设置动态分区分配选项卡"""
        layout = QVBoxLayout(tab)

        # 算法选择
        algo_group = QGroupBox("分配算法")
        algo_layout = QHBoxLayout()
        self.algo_combobox = QComboBox()
        self.algo_combobox.addItems(["首次适应算法", "最佳适应算法"])
        algo_layout.addWidget(QLabel("选择算法:"))
        algo_layout.addWidget(self.algo_combobox)
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # 操作按钮
        operations_group = QGroupBox("操作序列")
        operations_layout = QVBoxLayout()

        # 预定义操作序列
        operations = [
            "作业1申请130K", "作业2申请60K", "作业3申请100K",
            "作业2释放60K", "作业4申请200K", "作业3释放100K",
            "作业1释放130K", "作业5申请140K", "作业6申请60K",
            "作业7申请50K", "作业6释放60K"
        ]

        # 创建第一行按钮
        buttons_layout1 = QHBoxLayout()
        for i in range(0, min(5, len(operations))):
            btn = QPushButton(operations[i])
            # 使用functools.partial固定参数值，避免lambda陷阱
            btn.clicked.connect(lambda _, op=operations[i]: self._handle_operation(op))
            buttons_layout1.addWidget(btn)
        operations_layout.addLayout(buttons_layout1)

        # 创建第二行按钮
        buttons_layout2 = QHBoxLayout()
        for i in range(5, min(10, len(operations))):
            btn = QPushButton(operations[i])
            btn.clicked.connect(lambda _, op=operations[i]: self._handle_operation(op))
            buttons_layout2.addWidget(btn)
        operations_layout.addLayout(buttons_layout2)

        # 创建第三行按钮
        buttons_layout3 = QHBoxLayout()
        for i in range(10, len(operations)):
            btn = QPushButton(operations[i])
            btn.clicked.connect(lambda _, op=operations[i]: self._handle_operation(op))
            buttons_layout3.addWidget(btn)
        operations_layout.addLayout(buttons_layout3)

        operations_group.setLayout(operations_layout)
        layout.addWidget(operations_group)

        # 状态显示
        status_group = QGroupBox("内存状态")
        status_layout = QVBoxLayout()
        self.dynamic_status_text = QTextEdit()
        self.dynamic_status_text.setReadOnly(True)
        status_layout.addWidget(self.dynamic_status_text)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # 操作日志
        log_group = QGroupBox("操作日志")
        log_layout = QVBoxLayout()
        self.dynamic_log_text = QTextEdit()
        self.dynamic_log_text.setReadOnly(True)
        log_layout.addWidget(self.dynamic_log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # 重置按钮
        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self._reset_dynamic)
        layout.addWidget(reset_btn)

        # 初始化显示
        self._update_dynamic_status()

    def _setup_page_tab(self, tab):
        """设置页面置换选项卡"""
        layout = QVBoxLayout(tab)

        # 算法选择
        algo_group = QGroupBox("置换算法")
        algo_layout = QHBoxLayout()
        self.page_algo_combobox = QComboBox()
        self.page_algo_combobox.addItems(["FIFO", "LRU"])
        self.page_algo_combobox.currentTextChanged.connect(self._change_page_algorithm)
        algo_layout.addWidget(QLabel("选择算法:"))
        algo_layout.addWidget(self.page_algo_combobox)
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # 控制按钮
        control_group = QGroupBox("控制")
        control_layout = QHBoxLayout()

        generate_btn = QPushButton("生成指令序列")
        generate_btn.clicked.connect(self._generate_instructions)
        control_layout.addWidget(generate_btn)

        run_btn = QPushButton("运行")
        run_btn.clicked.connect(self._run_simulation)
        control_layout.addWidget(run_btn)

        self.step_btn = QPushButton("单步执行")
        self.step_btn.clicked.connect(self._step_simulation)
        control_layout.addWidget(self.step_btn)

        reset_page_btn = QPushButton("重置")
        reset_page_btn.clicked.connect(self._reset_page)
        control_layout.addWidget(reset_page_btn)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 指令序列显示
        instr_group = QGroupBox("指令序列")
        instr_layout = QVBoxLayout()
        self.instruction_text = QTextEdit()
        self.instruction_text.setReadOnly(True)
        instr_layout.addWidget(self.instruction_text)
        instr_group.setLayout(instr_layout)
        layout.addWidget(instr_group)

        # 执行状态显示
        exec_group = QGroupBox("执行状态")
        exec_layout = QVBoxLayout()
        self.exec_text = QTextEdit()
        self.exec_text.setReadOnly(True)
        exec_layout.addWidget(self.exec_text)
        exec_group.setLayout(exec_layout)
        layout.addWidget(exec_group)

        # 结果显示
        result_group = QGroupBox("结果")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        # 初始化变量
        self.instructions = []
        self.current_instruction_index = 0
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self._step_simulation)

    def _change_page_algorithm(self, algorithm):
        """更改页面置换算法"""
        self.current_algorithm = algorithm
        if algorithm == "FIFO":
            self.page_manager._replace_page = self.page_manager._replace_page_fifo
        else:  # LRU
            self.page_manager._replace_page = self.page_manager._replace_page_lru

        self.exec_text.append(f"已切换到{algorithm}页面置换算法")

    def _handle_operation(self, operation):
        """处理动态分区操作"""
        try:
            # 使用正则表达式解析操作字符串
            match = re.match(r"作业(\d+)(申请|释放)(\d+)?K", operation)
            if not match:
                self.dynamic_log_text.append(f"错误: 无效的操作格式: {operation}")
                return

            job_id = match.group(1)  # 提取作业ID
            action = match.group(2)  # 提取操作类型（"申请"或"释放"）
            size = int(match.group(3)) if action == "申请" else None  # 申请时提取大小

            if action == "申请":
                # 根据选择的算法分配内存
                if self.algo_combobox.currentText() == "首次适应算法":
                    result = self.dynamic_manager.allocate_first_fit(job_id, size)
                else:
                    result = self.dynamic_manager.allocate_best_fit(job_id, size)

                if result:
                    self.dynamic_log_text.append(f"成功: {operation}")
                else:
                    self.dynamic_log_text.append(f"失败: {operation} - 内存不足")
            else:
                # 释放内存
                result = self.dynamic_manager.free_memory(job_id)
                if result:
                    self.dynamic_log_text.append(f"成功: {operation}")
                else:
                    self.dynamic_log_text.append(f"失败: {operation} - 未找到该作业")

            self._update_dynamic_status()
        except Exception as e:
            self.dynamic_log_text.append(f"错误: 处理操作 {operation} 时发生异常: {str(e)}")
            import traceback
            traceback.print_exc()

    def _update_dynamic_status(self):
        """更新动态分区状态显示"""
        status = self.dynamic_manager.get_memory_status()
        self.dynamic_status_text.clear()
        for line in status:
            self.dynamic_status_text.append(line)

    def _reset_dynamic(self):
        """重置动态分区"""
        self.dynamic_manager = DynamicPartitionManager(640)
        self.dynamic_log_text.clear()
        self._update_dynamic_status()

    def _generate_instructions(self):
        """生成指令序列"""
        self.instructions = self.instruction_generator.generate_instructions()
        self.current_instruction_index = 0

        self.instruction_text.clear()
        for i, instr in enumerate(self.instructions):
            self.instruction_text.append(f"指令{i + 1}: 访问地址 {instr}")

        self.exec_text.clear()
        self.exec_text.append("已生成指令序列，可以开始执行")
        self.result_text.clear()

    def _run_simulation(self):
        """运行页面置换模拟"""
        if not self.instructions:
            self.exec_text.append("请先生成指令序列")
            return

        # 重置页面管理器
        self.page_manager.reset()
        self.current_instruction_index = 0
        self.exec_text.clear()

        # 启动定时器自动执行
        self.simulation_timer.start(100)  # 100ms执行一次

    def _step_simulation(self):
        """单步执行页面置换模拟"""
        if not self.instructions:
            self.exec_text.append("请先生成指令序列")
            return

        if self.current_instruction_index >= len(self.instructions):
            self.simulation_timer.stop()
            self._show_simulation_result()
            return

        # 执行当前指令
        instruction = self.instructions[self.current_instruction_index]
        result = self.page_manager.access_instruction(instruction)

        # 显示结果
        status = result["status"]
        page = result["page"]
        offset = result["offset"]

        if status == "命中":
            self.exec_text.append(f"指令{self.current_instruction_index + 1}: 访问地址 {instruction}, "
                                  f"页号 {page}, 偏移 {offset}, 命中, "
                                  f"物理地址 {result['physical_address']}")
        else:  # 缺页
            if result.get("replaced", False):
                self.exec_text.append(f"指令{self.current_instruction_index + 1}: 访问地址 {instruction}, "
                                      f"页号 {page}, 偏移 {offset}, 缺页, "
                                      f"置换页 {result['replaced_page']}, "
                                      f"物理地址 {result['physical_address']}")
            else:
                self.exec_text.append(f"指令{self.current_instruction_index + 1}: 访问地址 {instruction}, "
                                      f"页号 {page}, 偏移 {offset}, 缺页, "
                                      f"物理地址 {result['physical_address']}")

        self.current_instruction_index += 1

        # 如果已经执行完所有指令
        if self.current_instruction_index >= len(self.instructions):
            self.simulation_timer.stop()
            self._show_simulation_result()

    def _show_simulation_result(self):
        """显示页面置换模拟结果"""
        page_fault_rate = self.page_manager.get_page_fault_rate()
        self.result_text.clear()
        self.result_text.append(f"算法: {self.current_algorithm}")
        self.result_text.append(f"总指令数: {len(self.instructions)}")
        self.result_text.append(f"缺页次数: {self.page_manager.page_faults}")
        self.result_text.append(f"缺页率: {page_fault_rate:.4f}")

    def _reset_page(self):
        """重置页面置换"""
        self.simulation_timer.stop()
        self.page_manager.reset()
        self.current_instruction_index = 0
        self.instructions = []
        self.instruction_text.clear()
        self.exec_text.clear()
        self.result_text.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MemorySimulationApp()
    window.show()
    sys.exit(app.exec_())