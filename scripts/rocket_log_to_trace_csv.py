"""
Converts Rocketchip simulation log to riscv instruction trace format
"""

import argparse
import os
import re
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from riscv_trace_csv import *
from lib import *

# Rocket commit log format with embedded disassembly:
# 3 0x0080000000 (0xf14022f3) x 5 0x0000000000000000 csrr    t0,mhartid
# Special format: 3 0x0080000140 (0x03b45833) x16 p16 0xXXXXXXXXXXXXXXXX lw      a6, -4(a2)
# Also supports: 3 0x0080000148 (0x02244133) x 2 p 2 0xXXXXXXXXXXXXXXXX add     sp, sp, a2
# Also supports: 3 0x008000013a (0x00004101) x 2 0x0000000000000000 c.li    sp, 0
ROCKET_COMMIT_RE = re.compile(
    r"(?P<pri>\d+)\s+0x(?P<addr>[0-9a-fA-F]+)\s+\(0x(?P<bin>[0-9a-fA-F]+)\)(?:\s+(?P<reg>[xf])\s*(?P<reg_num>\d+)(?:\s+(?P<placeholder>p\s*\d+))?\s+0x(?P<val>[0-9a-fA-FX]+))?\s*(?P<disasm>.*?)$")

# Pattern for standalone register update lines: x16 p16 0x0000000000000001 or x 2 p 2 0xffffffffffffffff
STANDALONE_REG_RE = re.compile(
    r"(?P<reg>[xf])\s*(?P<reg_num>\d+)\s+(?P<placeholder>p\s*\d+)?\s+0x(?P<val>[0-9a-fA-F]+)")

ADDR_RE = re.compile(
    r"(?P<rd>[a-z0-9]+?),(?P<imm>[\-0-9]+?)\((?P<rs1>[a-z0-9]+)\)")

LOGGER = logging.getLogger()


def process_instr(trace):
    """Process instruction to normalize operand format"""
    if trace.instr == "jal":
        # Handle jal format
        idx = trace.operand.rfind(",")
        if idx != -1:
            imm = trace.operand[idx + 1:].strip()
            if imm.startswith("0x"):
                if imm[2] == "-":
                    imm = "-" + str(int(imm[3:], 16))
                else:
                    imm = str(int(imm[2:], 16))
                trace.operand = trace.operand[0:idx + 1] + imm
    
    # Convert "pc + N" or "pc - N" format to just "N" or "-N" for spike compatibility
    if "pc + " in trace.operand:
        trace.operand = trace.operand.replace("pc + ", "").replace(" ", "")
    elif "pc - " in trace.operand:
        trace.operand = trace.operand.replace("pc - ", "-").replace(" ", "")
    
    # Remove all spaces from operand for spike compatibility
    trace.operand = trace.operand.replace(" ", "")
    
    # Properly format operands of all instructions of the form:
    # <instr> <reg1> <imm>(<reg2>)
    # The operands should be converted into CSV as:
    # "<reg1>,<reg2>,<imm>"
    m = ADDR_RE.search(trace.operand)
    if m:
        trace.operand = "{},{},{}".format(
            m.group("rd"), m.group("rs1"), m.group("imm"))


def read_rocket_commit_instr(match, rv32=False):
    """Extract instruction info from Rocket commit log regex match"""
    
    pri = match.group("pri")
    addr = match.group("addr")
    binary = match.group("bin")
    reg_type = match.group("reg")
    reg_num = match.group("reg_num") 
    reg_val = match.group("val")
    disasm = match.group("disasm")  # Embedded disassembly
    
    instr = RiscvInstructionTraceEntry()
    # Convert to 64/32-bit address format
    num_char = 8 if rv32 else 16
    pc = addr[-num_char:] if len(addr) > num_char else addr.zfill(num_char)
    instr.pc = pc
    instr.binary = binary
    instr.mode = pri
    
    # Include register update if present and not register 0
    if reg_type and reg_num and reg_num != "0" and reg_val:
        if reg_type == "x":  # GPR
            reg_name = gpr_to_abi(f"x{reg_num}")
        elif reg_type == "f":  # FPR
            reg_name = f"f{reg_num}"
        else:
            reg_name = f"{reg_type}{reg_num}"
        
        # Convert to 64/32-bit register value
        reg_value = reg_val[-num_char:] if len(reg_val) > num_char else reg_val.zfill(num_char)
        instr.gpr.append(f"{reg_name}:{reg_value}")
    
    # Parse embedded disassembly
    if disasm and disasm.strip():
        disasm_clean = disasm.strip()
        
        # Parse instruction and operands from disassembly
        # Format: "auipc   a0, 0x0" or "csrr    t0,mhartid"
        parts = disasm_clean.split(None, 1)  # Split on whitespace, max 2 parts
        if len(parts) >= 1:
            instr.instr = parts[0].strip()
            instr.instr_str = disasm_clean
            
            if len(parts) >= 2:
                operand_str = parts[1].strip()
                
                # Handle PC-relative addressing
                if "pc + " in operand_str or "pc - " in operand_str:
                    # Convert PC-relative to absolute address for spike compatibility
                    current_pc = int(addr, 16)
                    if "pc + " in operand_str:
                        offset_str = operand_str.split("pc + ")[1].strip()
                        try:
                            offset = int(offset_str)
                            target_addr = current_pc + offset
                            operand_str = operand_str.replace(f"pc + {offset}", str(offset))
                        except ValueError:
                            pass  # Keep original if parsing fails
                    elif "pc - " in operand_str:
                        offset_str = operand_str.split("pc - ")[1].strip()
                        try:
                            offset = int(offset_str, 16) if offset_str.startswith('0x') else int(offset_str)
                            target_addr = current_pc - offset
                            operand_str = operand_str.replace(f"pc - {offset_str}", f"-{offset}")
                        except ValueError:
                            pass  # Keep original if parsing fails
                
                # Convert to spike-compatible operand format (remove spaces)
                operands = [op.strip() for op in operand_str.split(',') if op.strip()]
                instr.operand = ','.join(operands)
            else:
                instr.operand = ""
        else:
            instr.instr = "unknown"
            instr.instr_str = disasm_clean
            instr.operand = ""
    else:
        # No embedded disassembly available - fallback to unknown
        instr.instr_str = f"unknown_{binary}"
        instr.instr = "unknown"
        instr.operand = ""
    
    return instr


def read_rocket_commit_trace(path, rv32=False):
    """Read a Rocket commit log, yielding executed instructions.
    
    This function skips instructions until it reaches the entry point
    (PC 0x80000000) and then yields all committed instructions.
    Stops processing when it encounters an ecall instruction.
    Uses embedded disassembly from the rocket log for instruction information.
    """
    
    entry_point_reached = False
    entry_point = "80000000" if rv32 else "0000000080000000" 
    
    logging.info("Using embedded disassembly from rocket log")
    
    # Read all lines first to handle lookahead for special patterns
    with open(path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        match = ROCKET_COMMIT_RE.match(line)
        if not match:
            i += 1
            continue
            
        # Skip until we reach entry point
        addr = match.group("addr")
        num_char = 8 if rv32 else 16
        address = addr[-num_char:] if len(addr) > num_char else addr.zfill(num_char)
        
        if not entry_point_reached:
            if address.lower() == entry_point.lower():
                entry_point_reached = True
                # Don't continue here - process this instruction too
            else:
                i += 1
                continue
        
        # Check if this instruction has placeholder register values (0xXXXXXXXXXXXXXXXX)
        reg_val = match.group("val")
        if reg_val and 'X' in reg_val.upper():
            # Look for the correct value in the next few lines
            reg_type = match.group("reg")
            reg_num = match.group("reg_num")
            
            if reg_type and reg_num:
                # Search next few lines for standalone register update (expanded search range)
                for j in range(i + 1, min(i + 50, len(lines))):
                    next_line = lines[j].strip()
                    standalone_match = STANDALONE_REG_RE.match(next_line)
                    if (standalone_match and 
                        standalone_match.group("reg") == reg_type and 
                        standalone_match.group("reg_num") == reg_num):
                        # Replace the placeholder value with the correct one
                        correct_val = standalone_match.group("val")
                        logging.info(f"Replacing placeholder value for {reg_type}{reg_num} at PC {address}: {reg_val} -> {correct_val}")
                        # Create a new match dict with correct value
                        match_dict = match.groupdict()
                        match_dict["val"] = correct_val
                        # Create a simple object to mimic the match
                        class FixedMatch:
                            def __init__(self, groups):
                                self._groups = groups
                            def group(self, name):
                                return self._groups.get(name)
                        match = FixedMatch(match_dict)
                        break
        
        instr = read_rocket_commit_instr(match, rv32)
        
        # Check if this is an ecall instruction - if so, stop processing
        if instr.instr == "ecall":
            logging.info(f"Encountered ecall instruction at PC {instr.pc}, stopping trace processing")
            yield instr  # Include the ecall instruction in output
            break
        
        yield instr
        i += 1


def process_rocket_commit_log(rocket_log, csv, rv32=False):
    """Process Rocket commit log.
    
    Extract instruction and affected register information from Rocket commit log
    and write the results to a CSV file. Returns the number of instructions written.
    """
    logging.info("Processing Rocket commit log : {}".format(rocket_log))
    instrs_in = 0
    instrs_out = 0
    
    with open(csv, "w") as csv_fd:
        trace_csv = RiscvInstructionTraceCsv(csv_fd)
        trace_csv.start_new_trace()
        
        for entry in read_rocket_commit_trace(rocket_log, rv32):
            instrs_in += 1
            trace_csv.write_trace_entry(entry)
            instrs_out += 1
    
    logging.info("Processed instruction count : {}".format(instrs_in))
    logging.info("CSV saved to : {}".format(csv))
    return instrs_out


def main():
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, help="Input Rocket commit log")
    parser.add_argument("--csv", type=str, help="Output trace csv file")
    parser.add_argument("--rv32", dest="rv32", action="store_true",
                        help="RV32 mode (32-bit addresses)")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Verbose logging")
    parser.set_defaults(verbose=False, rv32=False)
    args = parser.parse_args()
    setup_logging(args.verbose)
    # Process Rocket commit log
    process_rocket_commit_log(args.log, args.csv, args.rv32)


if __name__ == "__main__":
    main()