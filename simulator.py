import sys

# Function to check and install required packages
def install_and_import(package):
    import importlib
    import subprocess
    import sys
    try:
        if package == 'ttk':
            import tkinter.ttk as ttk
        else:
            importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        globals()[package] = importlib.import_module(package)

# Check and import required packages
required_packages = ['tkinter', 'ttk', 'matplotlib', 'PyQt5', 'pyqtgraph']
for pkg in required_packages:
    install_and_import(pkg)

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import re

# Constants
NUM_REGISTERS = 4
MEMORY_SIZE = 256  # Adjusted for better display
REGISTER_NAMES = ['EAX', 'AX', 'AH', 'AL',
                  'EBX', 'BX', 'BH', 'BL',
                  'ECX', 'CX', 'CH', 'CL',
                  'EDX', 'DX', 'DH', 'DL']
MAIN_REGISTERS = ['EAX', 'EBX', 'ECX', 'EDX']
ADDRESSING_MODES = ["Immediate", "Register", "Direct", "Indirect", "Indexed"]
INSTRUCTIONS = ["MOV", "ADD", "SUB", "MUL", "DIV", "AND", "OR", "XOR", "NOT", "HALT"]

# Color Palette for Binary Segmentation
COLOR_PALETTE = {
    "opcode": "#FF5733",       # Orange
    "modrm": "#33FF57",        # Green
    "sib": "#3357FF",          # Blue
    "displacement": "#FF33A8", # Pink
    "immediate": "#FFFF00",    # Yellow
    "register": "#FFA500",     # Orange
    "memory": "#8A2BE2",       # BlueViolet
    "highlight": "#00FFFF",    # Cyan
}

# Descriptions for Binary Segments
SEGMENT_DESCRIPTIONS = {
    "opcode": "Opcode: Specifies the operation to perform.",
    "modrm": "ModR/M: Defines addressing mode and registers.",
    "sib": "SIB: Scale-Index-Base for complex addressing.",
    "displacement": "Displacement: Offset in memory addressing.",
    "immediate": "Immediate: Constant value in the instruction.",
    "register": "Register: Specifies the register involved.",
    "memory": "Memory: Indicates memory addressing.",
    "highlight": "Highlight: Used for special segments.",
}

class CPU:
    def __init__(self):
        # Main 32-bit registers
        self.registers = {'EAX': 0, 'EBX': 0, 'ECX': 0, 'EDX': 0}
        self.memory = [0] * MEMORY_SIZE
        self.PC = 0  # Program Counter
        self.halted = False
        # Flags
        self.ZF = 0  # Zero Flag
        self.CF = 0  # Carry Flag
        self.SF = 0  # Sign Flag
        self.OF = 0  # Overflow Flag
        # Instruction List
        self.instructions = []

    def reset(self):
        self.registers = {'EAX': 0, 'EBX': 0, 'ECX': 0, 'EDX': 0}
        self.memory = [0] * MEMORY_SIZE
        self.PC = 0
        self.halted = False
        self.ZF = 0
        self.CF = 0
        self.SF = 0
        self.OF = 0
        self.instructions = []

    def load_instructions(self, instructions):
        self.instructions = instructions
        self.PC = 0
        self.halted = False

    def get_register_value(self, reg_name):
        if reg_name == 'EAX':
            return self.registers['EAX'] & 0xFFFFFFFF
        elif reg_name == 'AX':
            return self.registers['EAX'] & 0xFFFF
        elif reg_name == 'AH':
            return (self.registers['EAX'] >> 8) & 0xFF
        elif reg_name == 'AL':
            return self.registers['EAX'] & 0xFF
        elif reg_name == 'EBX':
            return self.registers['EBX'] & 0xFFFFFFFF
        elif reg_name == 'BX':
            return self.registers['EBX'] & 0xFFFF
        elif reg_name == 'BH':
            return (self.registers['EBX'] >> 8) & 0xFF
        elif reg_name == 'BL':
            return self.registers['EBX'] & 0xFF
        elif reg_name == 'ECX':
            return self.registers['ECX'] & 0xFFFFFFFF
        elif reg_name == 'CX':
            return self.registers['ECX'] & 0xFFFF
        elif reg_name == 'CH':
            return (self.registers['ECX'] >> 8) & 0xFF
        elif reg_name == 'CL':
            return self.registers['ECX'] & 0xFF
        elif reg_name == 'EDX':
            return self.registers['EDX'] & 0xFFFFFFFF
        elif reg_name == 'DX':
            return self.registers['EDX'] & 0xFFFF
        elif reg_name == 'DH':
            return (self.registers['EDX'] >> 8) & 0xFF
        elif reg_name == 'DL':
            return self.registers['EDX'] & 0xFF
        else:
            raise ValueError(f"Invalid register name: {reg_name}")

    def set_register_value(self, reg_name, value):
        if reg_name == 'EAX':
            self.registers['EAX'] = value & 0xFFFFFFFF
        elif reg_name == 'AX':
            self.registers['EAX'] = (self.registers['EAX'] & 0xFFFF0000) | (value & 0xFFFF)
        elif reg_name == 'AH':
            self.registers['EAX'] = (self.registers['EAX'] & 0xFFFF00FF) | ((value & 0xFF) << 8)
        elif reg_name == 'AL':
            self.registers['EAX'] = (self.registers['EAX'] & 0xFFFFFF00) | (value & 0xFF)
        elif reg_name == 'EBX':
            self.registers['EBX'] = value & 0xFFFFFFFF
        elif reg_name == 'BX':
            self.registers['EBX'] = (self.registers['EBX'] & 0xFFFF0000) | (value & 0xFFFF)
        elif reg_name == 'BH':
            self.registers['EBX'] = (self.registers['EBX'] & 0xFFFF00FF) | ((value & 0xFF) << 8)
        elif reg_name == 'BL':
            self.registers['EBX'] = (self.registers['EBX'] & 0xFFFFFF00) | (value & 0xFF)
        elif reg_name == 'ECX':
            self.registers['ECX'] = value & 0xFFFFFFFF
        elif reg_name == 'CX':
            self.registers['ECX'] = (self.registers['ECX'] & 0xFFFF0000) | (value & 0xFFFF)
        elif reg_name == 'CH':
            self.registers['ECX'] = (self.registers['ECX'] & 0xFFFF00FF) | ((value & 0xFF) << 8)
        elif reg_name == 'CL':
            self.registers['ECX'] = (self.registers['ECX'] & 0xFFFFFF00) | (value & 0xFF)
        elif reg_name == 'EDX':
            self.registers['EDX'] = value & 0xFFFFFFFF
        elif reg_name == 'DX':
            self.registers['EDX'] = (self.registers['EDX'] & 0xFFFF0000) | (value & 0xFFFF)
        elif reg_name == 'DH':
            self.registers['EDX'] = (self.registers['EDX'] & 0xFFFF00FF) | ((value & 0xFF) << 8)
        elif reg_name == 'DL':
            self.registers['EDX'] = (self.registers['EDX'] & 0xFFFFFF00) | (value & 0xFF)
        else:
            raise ValueError(f"Invalid register name: {reg_name}")

    def execute_next_instruction(self):
        if self.halted or self.PC >= len(self.instructions):
            return "HALT"

        instr = self.instructions[self.PC]
        self.PC += 1
        parts = instr.strip().split()
        if not parts:
            return "NOP"

        opcode = parts[0].upper()

        if opcode == "HALT":
            self.halted = True
            return "HALT"

        if opcode not in INSTRUCTIONS:
            self.halted = True
            raise ValueError(f"Unsupported instruction: {opcode}")

        # Extract operands
        operands = ' '.join(parts[1:]).split(',')
        operands = [op.strip() for op in operands]

        # Execute based on opcode
        try:
            if opcode == "MOV":
                self.mov(operands)
            elif opcode == "ADD":
                self.add(operands)
            elif opcode == "SUB":
                self.sub(operands)
            elif opcode == "MUL":
                self.mul(operands)
            elif opcode == "DIV":
                self.div(operands)
            elif opcode == "AND":
                self.and_op(operands)
            elif opcode == "OR":
                self.or_op(operands)
            elif opcode == "XOR":
                self.xor_op(operands)
            elif opcode == "NOT":
                self.not_op(operands)
        except Exception as e:
            self.halted = True
            raise e

        return opcode

    def execute_all(self):
        while not self.halted and self.PC < len(self.instructions):
            opcode = self.execute_next_instruction()
            if opcode == "HALT":
                break

    def resolve_operand(self, operand):
        if operand.startswith("#"):  # Immediate
            try:
                value = int(operand[1:], 0)  # Supports decimal and hexadecimal
                return value, "Immediate"
            except ValueError:
                raise ValueError(f"Invalid immediate value: {operand}")
        elif operand in REGISTER_NAMES:  # Register
            value = self.get_register_value(operand)
            return value, "Register"
        elif operand.startswith("[") and operand.endswith("]"):  # Memory
            addr = operand[1:-1]
            if "+" in addr:
                # Indexed addressing [Base + Index*Scale + Disp]
                try:
                    base_part, rest = addr.split("+", 1)
                    index_part, disp = rest.split("+", 1)
                    index, scale = index_part.split("*")
                    base = base_part.strip()
                    index = index.strip()
                    scale = int(scale.strip())
                    disp = int(disp.strip())
                    if base not in MAIN_REGISTERS or index not in MAIN_REGISTERS:
                        raise ValueError(f"Invalid registers in indexed addressing: {operand}")
                    address = self.registers[base] + self.registers[index] * scale + disp
                except Exception:
                    raise ValueError(f"Invalid indexed addressing format: {operand}")
            else:
                # Direct or Indirect addressing
                addr = addr.strip()
                if addr in REGISTER_NAMES:
                    address = self.get_register_value(addr)
                else:
                    try:
                        address = int(addr, 0)  # Supports decimal and hexadecimal
                    except ValueError:
                        raise ValueError(f"Invalid memory address: {operand}")
            if address < 0 or address >= MEMORY_SIZE:
                raise ValueError(f"Memory address out of range: {address}")
            return self.memory[address], "Memory"
        else:
            raise ValueError(f"Invalid operand: {operand}")

    def set_operand(self, operand, value):
        if operand in REGISTER_NAMES:
            self.set_register_value(operand, value)
        elif operand.startswith("[") and operand.endswith("]"):
            addr = operand[1:-1]
            if "+" in addr:
                # Indexed addressing [Base + Index*Scale + Disp]
                try:
                    base_part, rest = addr.split("+", 1)
                    index_part, disp = rest.split("+", 1)
                    index, scale = index_part.split("*")
                    base = base_part.strip()
                    index = index.strip()
                    scale = int(scale.strip())
                    disp = int(disp.strip())
                    address = self.registers[base] + self.registers[index] * scale + disp
                except Exception:
                    raise ValueError(f"Invalid indexed addressing format: {operand}")
            else:
                addr = addr.strip()
                if addr in REGISTER_NAMES:
                    address = self.get_register_value(addr)
                else:
                    try:
                        address = int(addr, 0)  # Supports decimal and hexadecimal
                    except ValueError:
                        raise ValueError(f"Invalid memory address: {operand}")
            if address < 0 or address >= MEMORY_SIZE:
                raise ValueError(f"Memory address out of range: {address}")
            self.memory[address] = value & 0xFF  # Assuming memory is byte-addressable
        else:
            raise ValueError(f"Cannot set value to operand: {operand}")

    def update_flags(self, result, op1, op2, operation, bits):
        # ZF is set if result is zero
        self.ZF = int(result == 0)
        # SF is set if the result is negative (for signed integers)
        self.SF = int((result >> (bits - 1)) & 1)

        mask = (1 << bits) - 1

        if operation == 'ADD':
            # CF is set if there was a carry out
            self.CF = int((op1 + op2) > mask)
            # OF is set if there was a signed overflow
            op1_sign = (op1 >> (bits - 1)) & 1
            op2_sign = (op2 >> (bits - 1)) & 1
            res_sign = (result >> (bits - 1)) & 1
            self.OF = int((op1_sign == op2_sign) and (res_sign != op1_sign))
        elif operation == 'SUB':
            # CF is set if there was a borrow
            self.CF = int(op1 < op2)
            # OF is set if there was a signed overflow
            op1_sign = (op1 >> (bits - 1)) & 1
            op2_sign = (op2 >> (bits - 1)) & 1
            res_sign = (result >> (bits - 1)) & 1
            self.OF = int((op1_sign != op2_sign) and (res_sign != op1_sign))
        elif operation == 'MUL':
            # Simplified: set OF and CF if result exceeds the register size
            self.CF = self.OF = int(result > mask)
        elif operation == 'DIV':
            # Division doesn't affect CF and OF in this simplified model
            self.CF = self.OF = 0
        else:
            # For other operations, set CF and OF to 0
            self.CF = 0
            self.OF = 0

    # Instruction Implementations
    def mov(self, operands):
        if len(operands) != 2:
            raise ValueError("MOV requires two operands.")
        dest, src = operands
        src_val, src_type = self.resolve_operand(src)
        self.set_operand(dest, src_val)
        # Update flags based on the value moved
        bits = self.get_operand_size(dest)
        self.update_flags(src_val, 0, 0, 'MOV', bits)

    def add(self, operands):
        if len(operands) != 2:
            raise ValueError("ADD requires two operands.")
        dest, src = operands
        if dest not in REGISTER_NAMES:
            raise ValueError(f"Invalid destination register: {dest}")
        dest_val = self.get_register_value(dest)
        src_val, src_type = self.resolve_operand(src)
        result = dest_val + src_val

        bits = self.get_operand_size(dest)
        mask = (1 << bits) - 1
        result &= mask

        self.set_register_value(dest, result)
        self.update_flags(result, dest_val, src_val, 'ADD', bits)

    def sub(self, operands):
        if len(operands) != 2:
            raise ValueError("SUB requires two operands.")
        dest, src = operands
        if dest not in REGISTER_NAMES:
            raise ValueError(f"Invalid destination register: {dest}")
        dest_val = self.get_register_value(dest)
        src_val, src_type = self.resolve_operand(src)
        result = dest_val - src_val

        bits = self.get_operand_size(dest)
        mask = (1 << bits) - 1
        result &= mask

        self.set_register_value(dest, result)
        self.update_flags(result, dest_val, src_val, 'SUB', bits)

    def mul(self, operands):
        if len(operands) != 2:
            raise ValueError("MUL requires two operands.")
        dest, src = operands
        if dest not in REGISTER_NAMES:
            raise ValueError(f"Invalid destination register: {dest}")
        dest_val = self.get_register_value(dest)
        src_val, src_type = self.resolve_operand(src)
        result = dest_val * src_val

        bits = self.get_operand_size(dest)
        mask = (1 << bits) - 1
        result &= mask

        self.set_register_value(dest, result)
        self.update_flags(result, dest_val, src_val, 'MUL', bits)

    def div(self, operands):
        if len(operands) != 2:
            raise ValueError("DIV requires two operands.")
        dest, src = operands
        if dest not in REGISTER_NAMES:
            raise ValueError(f"Invalid destination register: {dest}")
        dest_val = self.get_register_value(dest)
        src_val, src_type = self.resolve_operand(src)
        if src_val == 0:
            raise ZeroDivisionError("Division by zero.")
        result = dest_val // src_val

        bits = self.get_operand_size(dest)
        mask = (1 << bits) - 1
        result &= mask

        self.set_register_value(dest, result)
        self.update_flags(result, dest_val, src_val, 'DIV', bits)

    def and_op(self, operands):
        if len(operands) != 2:
            raise ValueError("AND requires two operands.")
        dest, src = operands
        if dest not in REGISTER_NAMES:
            raise ValueError(f"Invalid destination register: {dest}")
        dest_val = self.get_register_value(dest)
        src_val, src_type = self.resolve_operand(src)
        result = dest_val & src_val

        bits = self.get_operand_size(dest)
        self.set_register_value(dest, result)
        self.update_flags(result, dest_val, src_val, 'AND', bits)

    def or_op(self, operands):
        if len(operands) != 2:
            raise ValueError("OR requires two operands.")
        dest, src = operands
        if dest not in REGISTER_NAMES:
            raise ValueError(f"Invalid destination register: {dest}")
        dest_val = self.get_register_value(dest)
        src_val, src_type = self.resolve_operand(src)
        result = dest_val | src_val

        bits = self.get_operand_size(dest)
        self.set_register_value(dest, result)
        self.update_flags(result, dest_val, src_val, 'OR', bits)

    def xor_op(self, operands):
        if len(operands) != 2:
            raise ValueError("XOR requires two operands.")
        dest, src = operands
        if dest not in REGISTER_NAMES:
            raise ValueError(f"Invalid destination register: {dest}")
        dest_val = self.get_register_value(dest)
        src_val, src_type = self.resolve_operand(src)
        result = dest_val ^ src_val

        bits = self.get_operand_size(dest)
        self.set_register_value(dest, result)
        self.update_flags(result, dest_val, src_val, 'XOR', bits)

    def not_op(self, operands):
        if len(operands) != 1:
            raise ValueError("NOT requires one operand.")
        dest = operands[0]
        if dest not in REGISTER_NAMES:
            raise ValueError(f"Invalid operand for NOT: {dest}")
        dest_val = self.get_register_value(dest)

        bits = self.get_operand_size(dest)
        mask = (1 << bits) - 1
        result = (~dest_val) & mask

        self.set_register_value(dest, result)
        self.update_flags(result, dest_val, 0, 'NOT', bits)

    def get_operand_size(self, reg_name):
        if reg_name in ['EAX', 'EBX', 'ECX', 'EDX']:
            return 32
        elif reg_name in ['AX', 'BX', 'CX', 'DX']:
            return 16
        elif reg_name in ['AH', 'AL', 'BH', 'BL', 'CH', 'CL', 'DH', 'DL']:
            return 8
        else:
            return 32  # Default to 32 bits

    def get_cpu_state(self):
        state = ""
        for reg in MAIN_REGISTERS:
            state += f"{reg}: {self.registers[reg]} (0x{self.registers[reg]:08X})\n"
        state += f"PC: {self.PC}\n"
        state += f"ZF: {self.ZF}, CF: {self.CF}, SF: {self.SF}, OF: {self.OF}"
        return state

    def get_memory_state(self):
        mem = ""
        for i in range(0, MEMORY_SIZE, 16):
            mem += f"{i:04X}: " + ' '.join(f"{byte:02X}" for byte in self.memory[i:i+16]) + "\n"
        return mem

class SimulatorGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.cpu = CPU()
        self.current_instruction = ""
        self.previous_registers = self.cpu.registers.copy()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("x86 Instruction Simulator")
        self.setGeometry(100, 100, 1600, 1000)  # Increased width for better layout

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Top Layout: Instruction Builder and Preview
        top_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top_layout)

        # Instruction Builder Frame
        builder_frame = QtWidgets.QGroupBox("Instruction Builder")
        builder_frame.setStyleSheet("QGroupBox { font-weight: bold; }")
        top_layout.addWidget(builder_frame, 2)
        builder_layout = QtWidgets.QVBoxLayout()
        builder_frame.setLayout(builder_layout)

        # Function Selection
        func_layout = QtWidgets.QHBoxLayout()
        builder_layout.addLayout(func_layout)
        func_label = QtWidgets.QLabel("Select Function:")
        func_label.setStyleSheet("font-weight: bold;")
        func_layout.addWidget(func_label)
        self.func_combo = QtWidgets.QComboBox()
        self.func_combo.addItems(INSTRUCTIONS)
        func_layout.addWidget(self.func_combo)
        self.func_combo.currentIndexChanged.connect(self.on_func_select)

        # Addressing Mode Selection
        addr_layout = QtWidgets.QHBoxLayout()
        builder_layout.addLayout(addr_layout)
        addr_label = QtWidgets.QLabel("Select Addressing Mode:")
        addr_label.setStyleSheet("font-weight: bold;")
        addr_layout.addWidget(addr_label)
        self.addr_combo = QtWidgets.QComboBox()
        self.addr_combo.addItems(ADDRESSING_MODES)
        addr_layout.addWidget(self.addr_combo)
        self.addr_combo.currentIndexChanged.connect(self.on_addr_select)

        # Operand Entry
        self.operand_layout = QtWidgets.QGridLayout()
        builder_layout.addLayout(self.operand_layout)
        self.setup_operands()

        # Instruction Preview
        preview_frame = QtWidgets.QGroupBox("Instruction Preview")
        preview_frame.setStyleSheet("QGroupBox { font-weight: bold; }")
        top_layout.addWidget(preview_frame, 1)
        preview_layout = QtWidgets.QVBoxLayout()
        preview_frame.setLayout(preview_layout)
        self.preview_label = QtWidgets.QLabel("")
        self.preview_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        preview_label_container = QtWidgets.QHBoxLayout()
        preview_label_container.addStretch()
        preview_label_container.addWidget(self.preview_label)
        preview_label_container.addStretch()
        preview_layout.addLayout(preview_label_container)

        # Middle Layout: Binary Visualization and Instruction List
        middle_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(middle_layout)

        # Binary Visualization with Color Legend
        visualization_container = QtWidgets.QWidget()
        visualization_layout = QtWidgets.QVBoxLayout()
        visualization_container.setLayout(visualization_layout)
        middle_layout.addWidget(visualization_container, 3)

        # Binary Visualization Plot
        self.visualization_widget = pg.GraphicsLayoutWidget()
        visualization_layout.addWidget(self.visualization_widget)
        self.binary_plot = self.visualization_widget.addPlot()
        self.binary_plot.hideAxis('left')
        self.binary_plot.hideAxis('bottom')

        # Color Legend
        legend_layout = QtWidgets.QHBoxLayout()
        visualization_layout.addLayout(legend_layout)
        legend_label = QtWidgets.QLabel("Color Legend:")
        legend_label.setStyleSheet("font-weight: bold;")
        legend_layout.addWidget(legend_label)
        for seg_type, color in COLOR_PALETTE.items():
            color_box = QtWidgets.QLabel()
            color_box.setFixedSize(20, 20)
            color_box.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
            text_label = QtWidgets.QLabel(f"{seg_type.capitalize()}")
            legend_layout.addWidget(color_box)
            legend_layout.addWidget(text_label)
        legend_layout.addStretch()

        # Instruction List
        instr_frame = QtWidgets.QGroupBox("Instruction List")
        instr_frame.setStyleSheet("QGroupBox { font-weight: bold; }")
        middle_layout.addWidget(instr_frame, 2)
        instr_layout = QtWidgets.QVBoxLayout()
        instr_frame.setLayout(instr_layout)
        self.instr_list = QtWidgets.QListWidget()
        self.instr_list.setStyleSheet("""
            QListWidget::item:selected {
                background-color: #ADD8E6;
            }
        """)
        instr_layout.addWidget(self.instr_list)

        # Control Buttons
        control_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(control_layout)
        add_button = QtWidgets.QPushButton("Add Instruction")
        add_button.setStyleSheet("padding: 10px; font-weight: bold;")
        add_button.clicked.connect(self.add_instruction)
        control_layout.addWidget(add_button)

        delete_button = QtWidgets.QPushButton("Delete Selected")
        delete_button.setStyleSheet("padding: 10px; font-weight: bold;")
        delete_button.clicked.connect(self.delete_instruction)
        control_layout.addWidget(delete_button)

        run_button = QtWidgets.QPushButton("Run")
        run_button.setStyleSheet("padding: 10px; font-weight: bold;")
        run_button.clicked.connect(self.run_instructions)
        control_layout.addWidget(run_button)

        step_button = QtWidgets.QPushButton("Step")
        step_button.setStyleSheet("padding: 10px; font-weight: bold;")
        step_button.clicked.connect(self.step_instruction)
        control_layout.addWidget(step_button)

        reset_button = QtWidgets.QPushButton("Reset")
        reset_button.setStyleSheet("padding: 10px; font-weight: bold;")
        reset_button.clicked.connect(self.reset_cpu)
        control_layout.addWidget(reset_button)

        # Bottom Layout: CPU State and Memory State
        bottom_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(bottom_layout)

        # CPU State
        cpu_frame = QtWidgets.QGroupBox("CPU State")
        cpu_frame.setStyleSheet("QGroupBox { font-weight: bold; }")
        bottom_layout.addWidget(cpu_frame, 2)
        cpu_layout = QtWidgets.QVBoxLayout()
        cpu_frame.setLayout(cpu_layout)
        self.state_table = QtWidgets.QTableWidget()
        self.state_table.setColumnCount(2)
        self.state_table.setHorizontalHeaderLabels(["Register", "Value"])
        self.state_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.state_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.state_table.horizontalHeader().setStretchLastSection(True)
        self.state_table.setStyleSheet("""
            QTableWidget {
                font-size: 14px;
            }
            QHeaderView::section {
                font-weight: bold;
                background-color: #D3D3D3;
            }
            QTableWidget::item:selected {
                background-color: #ADD8E6;
            }
        """)
        cpu_layout.addWidget(self.state_table)

        # Memory State
        mem_frame = QtWidgets.QGroupBox("Memory State")
        mem_frame.setStyleSheet("QGroupBox { font-weight: bold; }")
        bottom_layout.addWidget(mem_frame, 3)
        mem_layout = QtWidgets.QVBoxLayout()
        mem_frame.setLayout(mem_layout)
        self.memory_table = QtWidgets.QTableWidget()
        self.memory_table.setColumnCount(2)
        self.memory_table.setHorizontalHeaderLabels(["Address", "Data"])
        self.memory_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.memory_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.memory_table.horizontalHeader().setStretchLastSection(True)
        self.memory_table.setStyleSheet("""
            QTableWidget {
                font-size: 14px;
            }
            QHeaderView::section {
                font-weight: bold;
                background-color: #D3D3D3;
            }
            QTableWidget::item:selected {
                background-color: #ADD8E6;
            }
        """)
        mem_layout.addWidget(self.memory_table)

        self.on_func_select()

    def on_func_select(self):
        self.setup_operands()
        self.update_instruction_preview()

    def on_addr_select(self):
        self.setup_operands()
        self.update_instruction_preview()

    def setup_operands(self):
        # Clear existing operand widgets
        for i in reversed(range(self.operand_layout.count())):
            widget = self.operand_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        func = self.func_combo.currentText()
        mode = self.addr_combo.currentText()

        self.operands = []

        if func == "HALT":
            return
        elif func == "NOT":
            dest_label = QtWidgets.QLabel("Destination Register:")
            self.operand_layout.addWidget(dest_label, 0, 0)
            self.dest_combo = QtWidgets.QComboBox()
            self.dest_combo.addItems(REGISTER_NAMES)
            self.operand_layout.addWidget(self.dest_combo, 0, 1)
            self.dest_combo.currentIndexChanged.connect(self.update_instruction_preview)
            self.operands.append(self.dest_combo)
        else:
            dest_label = QtWidgets.QLabel("Destination Register:")
            self.operand_layout.addWidget(dest_label, 0, 0)
            self.dest_combo = QtWidgets.QComboBox()
            self.dest_combo.addItems(REGISTER_NAMES)
            self.operand_layout.addWidget(self.dest_combo, 0, 1)
            self.dest_combo.currentIndexChanged.connect(self.update_instruction_preview)
            self.operands.append(self.dest_combo)

            if mode == "Immediate":
                src_label = QtWidgets.QLabel("Immediate Value:")
                self.operand_layout.addWidget(src_label, 1, 0)
                self.src_input = QtWidgets.QLineEdit("#0")
                self.operand_layout.addWidget(self.src_input, 1, 1)
                self.src_input.textChanged.connect(self.update_instruction_preview)
                self.operands.append(self.src_input)
            elif mode == "Register":
                src_label = QtWidgets.QLabel("Source Register:")
                self.operand_layout.addWidget(src_label, 1, 0)
                self.src_combo = QtWidgets.QComboBox()
                self.src_combo.addItems(REGISTER_NAMES)
                self.operand_layout.addWidget(self.src_combo, 1, 1)
                self.src_combo.currentIndexChanged.connect(self.update_instruction_preview)
                self.operands.append(self.src_combo)
            elif mode == "Direct":
                src_label = QtWidgets.QLabel("Memory Address:")
                self.operand_layout.addWidget(src_label, 1, 0)
                self.src_input = QtWidgets.QLineEdit("0")
                self.operand_layout.addWidget(self.src_input, 1, 1)
                self.src_input.textChanged.connect(self.update_instruction_preview)
                self.operands.append(self.src_input)
            elif mode == "Indirect":
                src_label = QtWidgets.QLabel("Memory Register:")
                self.operand_layout.addWidget(src_label, 1, 0)
                self.src_combo = QtWidgets.QComboBox()
                self.src_combo.addItems(REGISTER_NAMES)
                self.operand_layout.addWidget(self.src_combo, 1, 1)
                self.src_combo.currentIndexChanged.connect(self.update_instruction_preview)
                self.operands.append(self.src_combo)
            elif mode == "Indexed":
                base_label = QtWidgets.QLabel("Base Register:")
                self.operand_layout.addWidget(base_label, 1, 0)
                self.base_combo = QtWidgets.QComboBox()
                self.base_combo.addItems(MAIN_REGISTERS)
                self.operand_layout.addWidget(self.base_combo, 1, 1)
                self.base_combo.currentIndexChanged.connect(self.update_instruction_preview)
                self.operands.append(self.base_combo)

                index_label = QtWidgets.QLabel("Index Register:")
                self.operand_layout.addWidget(index_label, 2, 0)
                self.index_combo = QtWidgets.QComboBox()
                self.index_combo.addItems(MAIN_REGISTERS)
                self.operand_layout.addWidget(self.index_combo, 2, 1)
                self.index_combo.currentIndexChanged.connect(self.update_instruction_preview)
                self.operands.append(self.index_combo)

                scale_label = QtWidgets.QLabel("Scale Factor:")
                self.operand_layout.addWidget(scale_label, 3, 0)
                self.scale_combo = QtWidgets.QComboBox()
                self.scale_combo.addItems(["1", "2", "4", "8"])
                self.operand_layout.addWidget(self.scale_combo, 3, 1)
                self.scale_combo.currentIndexChanged.connect(self.update_instruction_preview)
                self.operands.append(self.scale_combo)

                disp_label = QtWidgets.QLabel("Displacement:")
                self.operand_layout.addWidget(disp_label, 4, 0)
                self.disp_input = QtWidgets.QLineEdit("0")
                self.operand_layout.addWidget(self.disp_input, 4, 1)
                self.disp_input.textChanged.connect(self.update_instruction_preview)
                self.operands.append(self.disp_input)

    def update_instruction_preview(self):
        func = self.func_combo.currentText()
        mode = self.addr_combo.currentText()
        operand = ""

        try:
            if func == "HALT":
                instruction = "HALT"
            elif func == "NOT":
                dest = self.dest_combo.currentText()
                instruction = f"{func} {dest}"
            else:
                dest = self.dest_combo.currentText()
                if mode == "Immediate":
                    operand = self.src_input.text()
                    if not operand.startswith("#"):
                        operand = f"#{operand}"
                elif mode == "Register":
                    operand = self.src_combo.currentText()
                elif mode == "Direct":
                    operand = f"[{self.src_input.text()}]"
                elif mode == "Indirect":
                    operand = f"[{self.src_combo.currentText()}]"
                elif mode == "Indexed":
                    operand = f"[{self.base_combo.currentText()} + {self.index_combo.currentText()}*{self.scale_combo.currentText()} + {self.disp_input.text()}]"
                instruction = f"{func} {dest}, {operand}"

            # Basic syntax validation
            if func not in INSTRUCTIONS:
                raise ValueError(f"Unsupported instruction: {func}")

            self.preview_label.setText(instruction)
            self.current_instruction = instruction

            self.update_binary_visualization()
        except Exception as e:
            self.show_error_message(str(e), "Please check the instruction syntax and operands.")

    def update_binary_visualization(self):
        self.binary_plot.clear()
        if not self.current_instruction:
            return

        parts = self.current_instruction.split()
        if not parts:
            return

        opcode = parts[0]
        operands = ' '.join(parts[1:]).split(',') if len(parts) > 1 else []
        segments = []

        # Simulate x86 instruction encoding
        # Opcode (1 byte)
        opcode_bin = format(hash(opcode) % 256, '08b')
        segments.append((opcode_bin, "opcode", SEGMENT_DESCRIPTIONS["opcode"]))

        if operands:
            try:
                # Simulate ModR/M byte
                modrm_bin = format((hash(operands[0]) + hash(operands[1])) % 256, '08b')
                segments.append((modrm_bin, "modrm", SEGMENT_DESCRIPTIONS["modrm"]))

                # If Indexed addressing, include SIB byte
                if any(s in operands[1] for s in ['*', '+']):
                    sib_bin = format(hash('SIB') % 256, '08b')
                    segments.append((sib_bin, "sib", SEGMENT_DESCRIPTIONS["sib"]))

                # Simulate displacement and immediate values
                if '#' in operands[1]:
                    imm_value = operands[1].replace('#', '')
                    try:
                        imm_int = int(imm_value, 0)
                        immediate_bin = format(imm_int % 256, '08b')
                        segments.append((immediate_bin, "immediate", SEGMENT_DESCRIPTIONS["immediate"]))
                    except ValueError:
                        immediate_bin = '00000000'
                        segments.append((immediate_bin, "immediate", SEGMENT_DESCRIPTIONS["immediate"]))
                elif '[' in operands[1]:
                    # Extract displacement from memory operand
                    match = re.search(r'\+\s*([+-]?\d+)', operands[1])
                    if match:
                        disp = match.group(1)
                        try:
                            disp_int = int(disp, 0)
                            displacement_bin = format(disp_int % 256, '08b')
                            segments.append((displacement_bin, "displacement", SEGMENT_DESCRIPTIONS["displacement"]))
                        except ValueError:
                            displacement_bin = '00000000'
                            segments.append((displacement_bin, "displacement", SEGMENT_DESCRIPTIONS["displacement"]))
            except Exception as e:
                # Handle any unexpected errors during visualization
                self.show_error_message(str(e), "Error while encoding the instruction for visualization.")

        # Visualize the binary segments with descriptions
        x = 0
        for segment, seg_type, description in segments:
            color = COLOR_PALETTE.get(seg_type, "#000000")
            length = len(segment)
            rect = QtWidgets.QGraphicsRectItem(x, 0, length * 10, 50)
            rect.setBrush(pg.mkColor(color))
            rect.setPen(QtGui.QPen(QtCore.Qt.NoPen))
            self.binary_plot.addItem(rect)

            # Add text and tooltip
            text_item = pg.TextItem(segment, anchor=(0, 0))
            text_item.setPos(x, 0)
            text_item.setToolTip(description)  # Add description as a tooltip
            self.binary_plot.addItem(text_item)

            x += length * 10 + 10

    def add_instruction(self):
        instr = self.current_instruction.strip()
        if not instr:
            self.show_error_message("Empty Instruction", "Cannot add an empty instruction.")
            return
        self.instr_list.addItem(instr)
        self.current_instruction = ""
        self.preview_label.setText("")
        self.binary_plot.clear()

    def delete_instruction(self):
        selected_items = self.instr_list.selectedItems()
        if not selected_items:
            self.show_error_message("No Selection", "Please select an instruction to delete.")
            return
        for item in selected_items:
            self.instr_list.takeItem(self.instr_list.row(item))

    def run_instructions(self):
        instructions = [self.instr_list.item(i).text() for i in range(self.instr_list.count())]
        if not instructions:
            self.show_error_message("No Instructions", "There are no instructions to run.")
            return
        try:
            self.cpu.reset()
            self.previous_registers = self.cpu.registers.copy()
            self.cpu.load_instructions(instructions)
            self.cpu.execute_all()
            self.update_cpu_and_memory_state()
        except Exception as e:
            self.show_error_message(str(e), "An error occurred during instruction execution.")

    def step_instruction(self):
        if self.cpu.halted:
            self.show_error_message("CPU Halted", "The CPU is halted. Reset to continue.")
            return
        if self.cpu.PC >= len(self.cpu.instructions):
            self.show_error_message("End of Instructions", "No more instructions to execute.")
            return
        try:
            self.previous_registers = self.cpu.registers.copy()
            opcode = self.cpu.execute_next_instruction()
            self.update_cpu_and_memory_state()
        except Exception as e:
            self.show_error_message(str(e), "An error occurred during step execution.")

    def reset_cpu(self):
        self.cpu.reset()
        self.previous_registers = self.cpu.registers.copy()
        self.update_cpu_and_memory_state()
        self.instr_list.clear()
        self.state_table.clearContents()
        self.state_table.setRowCount(0)
        self.memory_table.clearContents()
        self.memory_table.setRowCount(0)
        self.binary_plot.clear()
        self.preview_label.setText("")

    def update_cpu_and_memory_state(self):
        self.state_table.clearContents()
        self.state_table.setRowCount(0)
        row = 0
        for reg in MAIN_REGISTERS:
            reg_value = self.cpu.registers[reg]
            sub_regs = {
                reg: (reg_value, 32),
                reg[1:]: (reg_value & 0xFFFF, 16),
                reg[1:] + 'H': ((reg_value >> 8) & 0xFF, 8),
                reg[1:] + 'L': (reg_value & 0xFF, 8)
            }
            for sub_reg, (value, bits) in sub_regs.items():
                self.state_table.insertRow(row)
                self.state_table.setItem(row, 0, QtWidgets.QTableWidgetItem(sub_reg))
                item = QtWidgets.QTableWidgetItem(f"{value} (0x{value:0{bits//4}X})")
                if self.previous_registers[reg] != self.cpu.registers[reg]:
                    item.setBackground(QtGui.QColor('yellow'))
                self.state_table.setItem(row, 1, item)
                row += 1

        flags = [("PC", self.cpu.PC), ("ZF", self.cpu.ZF), ("CF", self.cpu.CF), ("SF", self.cpu.SF), ("OF", self.cpu.OF)]
        for flag, val in flags:
            self.state_table.insertRow(row)
            self.state_table.setItem(row, 0, QtWidgets.QTableWidgetItem(flag))
            self.state_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(val)))
            row += 1

        self.memory_table.clearContents()
        self.memory_table.setRowCount(0)
        for i in range(0, MEMORY_SIZE, 16):
            addr = f"{i:04X}"
            mem_values = ' '.join(f"{byte:02X}" for byte in self.cpu.memory[i:i+16])
            row_idx = self.memory_table.rowCount()
            self.memory_table.insertRow(row_idx)
            self.memory_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(addr))
            self.memory_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(mem_values))

        self.previous_registers = self.cpu.registers.copy()

    def show_error_message(self, message, suggestion):
        error_dialog = QtWidgets.QMessageBox(self)
        error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.setInformativeText(suggestion)
        error_dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
        error_dialog.exec_()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    simulator = SimulatorGUI()
    simulator.show()
    sys.exit(app.exec_())
