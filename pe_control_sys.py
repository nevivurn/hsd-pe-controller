from amaranth import *
from pe_stack import PEStack
from enum import IntEnum
import math
from amaranth.lib.fifo import SyncFIFOBuffered
from or_tree import ORTree

# general design

# n by n PE control
####################
#          W / B
#      -------------
#     |             |
#  A  |    n x n    |
#     |             |
#      -------------
#

# instruction set
# OP     V1               V2
# 4b     4b               24b
# load   to[a|b]          from(gb_addr)
# exec   {1b FIFO flow/reuse, 1b activation function code}
#                         value_for_activation # feed until queue is empty
# store                   to(gb_addr)
# flush
# set_m                   {sys_h, sys_w, fan_in}


# memory layout before start
# 0xCAFE0000  # start of code. tell if input is ready or not
# ...         # instructions, always start at 1
# 0xCAFECAFE  # end of code

# memory layout during & after execution
# 0xCAFENNNN  # NNNN is number of stores done

# load
# load from global buffer to [a|b] local buffer
# global buffer address start from gb_addr
# local buffer address always starts from 0
# size is defined by set_m

# store
# store from PE buffer to global buffer
# global buffer address start from gb_addr
# size is defined by set_m

# exec
# execute for fan_in cycles
# V1 is as follows:
# MSB                              LSB
# 1b    1b                   1b    1b
# nop   FIFO B flow/reuse    nop   activation function code

# activation function code
# 0 for None
# 1 for ReLU

# set_m
# store fan_in, sys_w, sys_h in specified register

# states
# INIT
# FETCH
# DECODE  # also handle set_m
# LOAD
# EXEC
# FLUSH
# STORE


# 4-bit OPCode
class OPCODE(IntEnum):
    LOAD = 0
    EXEC = 1
    STORE = 2
    FLUSH = 3
    SET_M = 4


# 4-bit LOAD destination
class LOAD_DEST(IntEnum):
    A = 0
    W = 1


# 1-bit activation code
class ACTCODE(IntEnum):
    NONE = 0
    RELU = 1


# 1-bit FIFO B flow/reuse
class FIFO(IntEnum):
    FLOW = 0
    REUSE = 1


class PEControl(Elaboratable):
    def __init__(
            self, num_bits, width, g_depth, l_depth,
            sys_size_bit, signed=True):
        self.num_bits = num_bits
        self.acc_bits = width
        self.width = width  # global buffer, local buffer width (bits per line)
        self.g_depth = g_depth  # global buffer depth (number of lines)
        self.l_depth = l_depth  # local buffer depth (number of lines)
        self.sys_size_bit = sys_size_bit
        self.sys_size = 2 ** sys_size_bit
        self.signed = signed
        self.next_pc_bits = 8

        assert width in [32, 64, 128]
        # BRAM constraint of Virtex-6 devices
        assert l_depth in [1024, 2048, 4096, 8192, 16384]
        assert g_depth in [1024, 2048, 4096, 8192, 16384]
        assert l_depth <= g_depth

        self.gb_addr_width = int(round(math.log2(g_depth)))
        self.cnt_bits = int(round(math.log2(l_depth)))

        assert self.cnt_bits + 2 * sys_size_bit <= 24
        assert self.gb_addr_width >= self.next_pc_bits

        # reset signal
        self.in_rst = Signal(1, reset_less=True)

        # global buffer interface
        self.in_r_data = Signal(width)
        self.out_r_addr = Signal(self.gb_addr_width)
        self.out_w_en = Signal(1)
        self.out_w_data = Signal(width)
        self.out_w_addr = Signal(self.gb_addr_width)

        # global buffer fused address
        self.addr_io = Signal(self.gb_addr_width)
        self.addr_io_ovf = Signal(1)

        # n by n PE stack
        self.pe = Array([
            Array([
                PEStack(
                    num_bits=num_bits, width=width,
                    cnt_bits=self.cnt_bits, signed=signed)
                for j in range(self.sys_size)])
            for i in range(self.sys_size)])

        # n buffer_a + n buffer_b
        self.lb_a = Array(
            [SyncFIFOBuffered(width=width, depth=l_depth + 1)
             for i in range(self.sys_size)])
        self.lb_b = Array(
            [SyncFIFOBuffered(width=width, depth=l_depth + 1)
             for i in range(self.sys_size)])

        # next program counter
        self.next_pc = Signal(self.next_pc_bits, reset=1)
        self.next_pc_ovf = Signal(1)

        self.opcode = Signal(4)
        self.v1 = Signal(4)
        self.v2 = Signal(24)

        self.v2_f = Signal(self.cnt_bits)
        self.v2_h = Signal(self.sys_size_bit)
        self.v2_w = Signal(self.sys_size_bit)

        # control fan in
        self.fan_in = Signal(self.cnt_bits)  # storage
        self.fan_cnt = Signal(self.cnt_bits)  # count
        self.fan_cnt_ovf = Signal(1)
        self.fan_cnt_next = Signal(self.cnt_bits + 1)

        self.is_store = Signal(1)

        self.reuse_b = Signal(1)
        self.sel_a = Signal(1)
        self.sel_b = Signal(1)
        self.act_code = Signal(1)

        # 0xCAFE_NNNN where NNNN is number of stores done
        self.magic_cnt = Signal(32)
        self.magic_cnt_ovf = Signal(1)
        self.magic_init = 0xCAFE0000

        self.pe_init = Signal(self.cnt_bits)

        self.lb_a_r_en = Signal(1)
        self.lb_b_r_en = Signal(1)
        self.lb_a_w_en = Signal(1)
        self.lb_b_w_en = Signal(1)

        self.sys_h = Signal(self.sys_size_bit)
        self.sys_w = Signal(self.sys_size_bit)

        self.sys_sel_row = Signal(self.sys_size)
        self.sys_sel_col = Signal(self.sys_size)

        self.sys_h_sel = Signal(self.sys_size)
        self.sys_w_sel = Signal(self.sys_size)

        self.row_ortree = Array([
            ORTree(num_bits=self.acc_bits, fan_in=self.sys_size)
            for i in range(self.sys_size)
        ])
        self.out_ortree = ORTree(num_bits=self.acc_bits, fan_in=self.sys_size)
        self.pe_star = Signal(width)

        self.lb_b_rdy_ortree = ORTree(num_bits=1, fan_in=self.sys_size)
        self.lb_b_rdy_any = Signal(1)

    def elaborate(self, platform):
        m = Module()

        m.submodules.out_ortree = self.out_ortree
        m.submodules.lb_b_rdy_ortree = self.lb_b_rdy_ortree

        for i in range(self.sys_size):
            setattr(m.submodules, f'pipe_a_{i}', self.lb_a[i])
            setattr(m.submodules, f'pipe_b_{i}', self.lb_b[i])
            setattr(m.submodules, f'row_ortree_{i}', self.row_ortree[i])

            m.d.comb += [
                self.lb_a[i].r_en.eq(self.lb_a_r_en),
                self.lb_b[i].r_en.eq(self.lb_b_r_en),

                self.lb_a[i].w_en.eq(self.lb_a_w_en & self.sys_sel_row[i]),
                self.lb_b[i].w_en.eq(self.lb_b_w_en & self.sys_sel_col[i]),

                self.lb_a[i].w_data.eq(self.in_r_data),
                self.lb_b[i].w_data.eq(Mux(self.reuse_b, self.lb_b[i].r_data, self.in_r_data)),

                self.out_ortree.in_arr[i].eq(self.row_ortree[i].out_d),
                self.lb_b_rdy_ortree.in_arr[i].eq(self.lb_b[i].r_rdy),
            ]
            for j in range(self.sys_size):
                setattr(m.submodules, f'pe_{i}_{j}', self.pe[i][j])

                m.d.comb += [
                    self.pe[i][j].in_rst.eq(self.in_rst),
                    self.pe[i][j].in_a.eq(self.lb_a[i].r_data),
                    self.pe[i][j].in_b.eq(self.lb_b[j].r_data),
                    self.pe[i][j].in_init.eq(self.pe_init),

                    self.row_ortree[i].in_arr[j].eq(
                        Mux(self.sys_sel_row[i] & self.sys_sel_col[j],
                            self.pe[i][j].out_d,
                            0)
                    ),
                ]

        m.d.comb += [
            self.out_r_addr.eq(self.addr_io),
            self.out_w_addr.eq(self.addr_io),
            Cat(self.v2, self.v1, self.opcode).eq(self.in_r_data[:32]),
            self.fan_cnt_next.eq(self.fan_cnt + 1),
            Cat(self.v2_f, self.v2_w, self.v2_h).eq(
                self.v2[:self.cnt_bits+2*self.sys_size_bit]
            ),
            self.pe_star.eq(self.out_ortree.out_d),
            self.out_w_data.eq(
                Mux(~self.is_store, self.magic_cnt, self.pe_star)
            ),
            self.sys_h_sel.eq(1 << self.sys_h),
            self.sys_w_sel.eq(1 << self.sys_w),
            self.lb_b_rdy_any.eq(self.lb_b_rdy_ortree.out_d),
            # feel free to add any combinatorial logics
        ]

        with m.FSM(reset='INIT'):
            with m.State('INIT'):
                m.d.sync += [
                    Cat(self.addr_io, self.addr_io_ovf).eq(0),
                    Cat(self.magic_cnt, self.magic_cnt_ovf).eq(self.magic_init),
                ]

                with m.If(self.in_r_data == self.magic_init):
                    Cat(self.next_pc, self.next_pc_ovf).eq(1)
                    m.next = 'FETCH'

            with m.State('FETCH'):
                m.d.sync += [
                    Cat(self.addr_io, self.addr_io_ovf).eq(self.next_pc),
                    self.out_w_en.eq(0),

                    self.lb_a_w_en.eq(0),
                    self.lb_b_w_en.eq(0),
                    self.lb_a_r_en.eq(0),
                    self.lb_b_r_en.eq(0),
                ]
                with m.If(self.addr_io == self.next_pc):
                    m.d.sync += Cat(self.next_pc, self.next_pc_ovf).eq(self.next_pc+1),
                    m.next = 'DECODE'

            with m.State('DECODE'):
                with m.Switch(self.opcode):
                    with m.Case(OPCODE.LOAD):
                        # OP     V1               V2
                        # 4b     4b               24b
                        # load   to[a|b]          from(gb_addr)
                        m.d.sync += [
                            Cat(self.addr_io, self.addr_io_ovf).eq(self.v2),
                            self.sel_a.eq(~self.v1),
                            self.sel_b.eq(self.v1),
                            self.sys_sel_row.eq(1),
                            self.sys_sel_col.eq(1),
                            Cat(self.fan_cnt, self.fan_cnt_ovf).eq(0),
                        ]
                        m.next = 'LOAD'

                    with m.Case(OPCODE.EXEC):
                        # OP     V1               V2
                        # 4b     4b               24b
                        # exec   {2b FIFO flush/reuse, 2b activation f code}
                        #                         value_for_activation
                        # 2b                   2b
                        # FIFO flush/reuse     activation function code

                        # FIFO flush/reuse
                        # 0   : flush   1   : reuse
                        # MSB : a       LSB : b

                        # activation function code
                        # 0 for None
                        # 1 for ReLU
                        m.next = 'EXEC'
                        m.d.sync += [
                            Cat(self.fan_cnt, self.fan_cnt_ovf).eq(1),
                            self.reuse_b.eq(self.v1[2]),
                            self.pe_init.eq(self.fan_in),

                            self.sys_sel_row.eq((1<<self.sys_size)-1),
                            self.sys_sel_col.eq((1<<self.sys_size)-1),
                        ]

                    with m.Case(OPCODE.STORE):
                        # OP     V1               V2
                        # 4b     4b               24b
                        # store                   to(gb_addr)
                        m.next = 'STORE'
                        m.d.sync += [
                            self.out_w_en.eq(1),
                            Cat(self.addr_io, self.addr_io_ovf).eq(self.v2),

                            Cat(self.magic_cnt, self.magic_cnt_ovf).eq(self.magic_cnt + 1),
                            self.is_store.eq(1),

                            self.sys_sel_row.eq(1),
                            self.sys_sel_col.eq(1),
                        ]

                    with m.Case(OPCODE.FLUSH):
                        m.next = 'FLUSH'
                        m.d.sync += [
                            self.lb_b_r_en.eq(1),
                        ]

                    with m.Case(OPCODE.SET_M):
                        # OP     V1               V2
                        # 4b     4b               24b
                        # set_m                   {sys_h, sys_w, fan_in}
                        # NOTE 0-base vs. 1-base
                        # for example,
                        # fan_in == 0 --> invalid
                        # fan_in == 1 --> scalar product
                        # sys_h == 0 and sys_w == 0 --> 1 by 1 PE control
                        m.next = 'FETCH'
                        m.d.sync += [
                            self.fan_in.eq(Mux(self.v2_f.any(), self.v2_f, self.fan_in)),
                            self.sys_w.eq(self.v2_w),
                            self.sys_h.eq(self.v2_h),
                        ]

                    with m.Case():  # default
                        m.next = 'INIT'

            with m.State('LOAD'):
                ###################################################
                # FROM
                # load to one buffer
                ###################################################
                # TO
                # load to multiple buffers serially
                ###################################################
                m.d.sync += [
                    Cat(self.addr_io, self.addr_io_ovf).eq(self.addr_io + 1),
                    Cat(self.fan_cnt, self.fan_cnt_ovf).eq(self.fan_cnt_next),

                    self.lb_a_w_en.eq(self.sel_a),
                    self.lb_b_w_en.eq(self.sel_b),
                ]
                with m.If(self.fan_cnt == self.fan_in):
                    m.d.sync += [
                        self.sys_sel_row.eq(self.sys_sel_row.rotate_left(1)),
                        self.sys_sel_col.eq(self.sys_sel_col.rotate_left(1)),
                        Cat(self.fan_cnt, self.fan_cnt_ovf).eq(1),
                    ]
                    with m.If(Mux(self.sel_a, self.sys_sel_row == self.sys_h_sel, self.sys_sel_col == self.sys_w_sel)):
                        m.next = 'FETCH'
                        m.d.sync += [
                            self.lb_a_w_en.eq(0),
                            self.lb_b_w_en.eq(0),
                        ]

            with m.State('EXEC'):
                m.d.sync += [
                    self.pe_init.eq(0),

                    Cat(self.fan_cnt, self.fan_cnt_ovf).eq(self.fan_cnt_next),
                    self.lb_b_w_en.eq(self.reuse_b),

                    self.lb_a_r_en.eq(1),
                    self.lb_b_r_en.eq(1),
                ]

                with m.If(self.fan_cnt == self.fan_in):
                    m.next = 'FETCH'

            with m.State('STORE'):
                ###################################################
                # FROM
                # store one result from one PE
                ###################################################
                # TO
                # store multiple results from sys_h by sys_w PEs
                ###################################################
                m.d.sync += [
                    Cat(self.addr_io, self.addr_io_ovf).eq(self.addr_io + 1),
                    self.sys_sel_col.eq(self.sys_sel_col.rotate_left(1)),
                ]
                with m.If(self.sys_sel_col == self.sys_w_sel):
                    m.d.sync += [
                        self.sys_sel_col.eq(1),
                        self.sys_sel_row.eq(self.sys_sel_row.rotate_left(1)),
                    ]
                    with m.If(self.sys_sel_row == self.sys_h_sel):
                        m.d.sync += [
                            Cat(self.addr_io, self.addr_io_ovf).eq(0),
                            self.is_store.eq(0),
                        ]
                        m.next = 'FETCH'

            with m.State('FLUSH'):
                with m.If(~self.lb_b_rdy_any):  # all empty
                    m.next = 'FETCH'
                    m.d.sync += [
                        self.lb_b_r_en.eq(0),
                    ]

        return m


if __name__ == '__main__':
    width = 32
    unit_width = 8
    signed = True
    sys_size_bit = 2
    g_depth = 1024
    l_depth = 1024

    unit_width_mask = (1 << unit_width) - 1

    cnt_bits = int(round(math.log2(l_depth)))

    print(f'systolic array of size {2**sys_size_bit} x {2**sys_size_bit}')
    dut = PEControl(unit_width, width,
                    g_depth=g_depth, l_depth=l_depth,
                    sys_size_bit=sys_size_bit, signed=signed)
    dut = ResetInserter(dut.in_rst)(dut)

    from amaranth.sim import Simulator
    import numpy as np
    from collections import deque

    np.random.seed(42)

    def test_case(dut, in_r_data):
        yield dut.in_r_data.eq(int(in_r_data))
        yield
        w_data = yield dut.out_w_data
        return w_data

    def make_code(a, b=0, c=0):
        return (a << 28) + (b << 24) + c

    # NOTE all operations cover timeframe from DECODE to FETCH
    def initialize():
        # INIT(1) + FETCH(2)
        for _ in range(3):
            yield from test_case(dut, in_r_data=0xCAFE0000)

    def load(to, data_arr, fan_in, start_addr):
        code = make_code(OPCODE.LOAD, to, start_addr)
        # DECODE(1) + LOAD(1, memory read delay)
        for _ in range(2):
            yield from test_case(dut, in_r_data=code)

        # LOAD(fan_in) + FETCH(2)
        for _ in range(fan_in):
            yield from test_case(dut, in_r_data=data_arr.popleft())
        for _ in range(2):
            yield from test_case(dut, in_r_data=0xdeadbeef)

    def set_metadata(fan_in, sys_h, sys_w):
        code = make_code(
            OPCODE.SET_M, 0, (sys_h << (cnt_bits + sys_size_bit)) |
            (sys_w << cnt_bits) | fan_in)
        # DECODE(1) + FETCH(2)
        for _ in range(3):
            yield from test_case(dut, in_r_data=code)

    def execute(fan_in, reuse=False):
        v2 = ACTCODE.NONE
        if reuse:
            v2 |= (FIFO.REUSE << 2)
        code = make_code(OPCODE.EXEC, v2, 0)
        # DECODE(1) + EXEC(fan_in) + FETCH(2)
        for _ in range(fan_in + 3):
            yield from test_case(dut, in_r_data=code)

    def flush_lb(fan_in_prev=0):
        code = make_code(OPCODE.FLUSH)
        # DECODE(1) + FLUSH(fan_in_prev) + FETCH(2)
        if fan_in_prev == 0:
            fan_in_prev += 1  # at least one cycle to check if queue is empty``
        for _ in range(fan_in_prev + 3):
            yield from test_case(dut, in_r_data=code)

    def store(sys_h, sys_w, start_addr):
        code = make_code(
            OPCODE.STORE, 0, start_addr)

        store_data = []

        # DECODE(1) + STORE(height * width)
        for _ in range(1 + (sys_h + 1) * (sys_w + 1)):
            w_data = yield from test_case(dut, in_r_data=code)
            if w_data >> (width - 1):
                w_data -= (1 << width)
            store_data.append(w_data)
        # FETCH(2)
        yield from test_case(dut, in_r_data=0xff)
        yield from test_case(dut, in_r_data=0xCAFE0000 + 1)

        return store_data[1:]

    def handle_overflow(d, bit=unit_width):
        sign_max = 2 ** (bit - 1) - 1
        sign_min = -(2 ** (bit - 1))
        if d > sign_max:
            d -= (2 ** bit)
        elif d < sign_min:
            d += (2 ** bit)
        return d

    def vec_to_mat(data, fan_in):
        mat = []
        row = []
        for i, d in enumerate(data):
            for j in range(width // unit_width):
                mask = unit_width_mask << (j * unit_width)
                row.append(
                    handle_overflow((d & mask) >> (j * unit_width), unit_width)
                )
            if (i+1) % fan_in == 0:
                mat.append(row)
                row = []
        return np.array(mat)

    def bench():
        num_runs = 8
        is_first = None

        lb_a = deque()
        lb_b = deque()

        for i in range(num_runs):
            is_first = i % 2 == 0
            if is_first:
                fan_in = np.random.randint(low=0, high=16) % 4 + 1
                sys_h = np.random.randint(low=0, high=2**sys_size_bit)
                sys_w = np.random.randint(low=0, high=2**sys_size_bit)
                data_b = np.random.randint(
                    low=0, high=2**width, size=[fan_in * (sys_w + 1)])
                for d in data_b:
                    lb_b.append(d)
            print(f'settings : {sys_h+1} x {sys_w+1}')
            data_a = np.random.randint(
                low=0, high=2**width, size=[fan_in * (sys_h + 1)])
            for d in data_a:
                lb_a.append(d)

            mat_a = vec_to_mat(data_a, fan_in)
            mat_b = vec_to_mat(data_b, fan_in)
            result = np.matmul(mat_a, np.transpose(mat_b)).reshape([-1])
            print(result)

            if is_first:
                yield from initialize()
                yield from set_metadata(fan_in, sys_h, sys_w)
            yield from load(LOAD_DEST.A, lb_a, fan_in * (sys_h + 1),
                            start_addr=8)
            if is_first:
                yield from load(LOAD_DEST.W, lb_b, fan_in * (sys_w + 1),
                                start_addr=12)
            yield from execute(fan_in, reuse=is_first)

            if not is_first:
                yield from flush_lb(len(lb_b))

            store_data = yield from store(sys_h, sys_w, start_addr=16)
            print(store_data)
            assert np.allclose(result, store_data)

            # end of call
            if not is_first:
                for _ in range(1):
                    yield from test_case(dut, in_r_data=0xCAFE0000 + 1)

    from amaranth.sim import Simulator

    from pathlib import Path
    p = Path(__file__)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_sync_process(bench)

    with open(p.with_suffix('.vcd'), 'w') as f:
        with sim.write_vcd(f):
            sim.run()

    from amaranth.back import verilog
    top = PEControl(unit_width, width,
                    g_depth=g_depth, l_depth=l_depth,
                    sys_size_bit=sys_size_bit, signed=signed)
    with open(p.with_suffix('.v'), 'w') as f:
        f.write(
            verilog.convert(
                top,
                ports=[top.in_r_data, top.out_r_addr,
                       top.out_w_addr, top.out_w_en, top.out_w_data]))
