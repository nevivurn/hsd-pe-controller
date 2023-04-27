from amaranth import *
from amaranth.lib.fifo import SyncFIFOBuffered


class TestFIFO(Elaboratable):
    def __init__(self, width, depth):
        self.pipe = SyncFIFOBuffered(width=width, depth=depth)

        self.in_data = Signal(width)
        self.out_w_rdy = Signal(1)
        self.in_w_en = Signal(1)
        self.out_w_level = Signal(range(depth + 1))

        self.out_data = Signal(width)
        self.out_r_rdy = Signal(1)
        self.in_r_en = Signal(1)
        self.out_r_level = Signal(range(depth + 1))

    def elaborate(self, platform):
        m = Module()

        m.submodules.pipe = pipe = self.pipe

        m.d.comb += [
            self.out_w_rdy.eq(pipe.w_rdy),
            self.out_w_level.eq(pipe.w_level),
            self.out_r_rdy.eq(pipe.r_rdy),
            self.out_r_level.eq(pipe.r_level),
            self.out_data.eq(pipe.r_data),

            pipe.w_data.eq(self.in_data),
            pipe.w_en.eq(self.in_w_en),
            pipe.r_en.eq(self.in_r_en),
        ]

        return m

# w_data : Signal(width), in
# w_rdy : Signal(1), out
# w_en : Signal(1), in
# w_level : Signal(range(depth + 1)), out

# r_data : Signal(width), out
# r_rdy : Signal(1), out
# r_en : Signal(1), in
# r_level : Signal(range(depth + 1)), out


if __name__ == '__main__':
    width = 8
    # NOTE fifo of with depth `n`, write `n` and read `n`
    # n = 2 --> FAIL
    # n > 2 --> PASS
    depth = 3
    dut = TestFIFO(width=width, depth=depth)

    from amaranth.sim import Simulator

    def test_case(dut, in_data, in_w_en, in_r_en):
        yield dut.in_data.eq(in_data)
        yield dut.in_w_en.eq(in_w_en)
        yield dut.in_r_en.eq(in_r_en)
        yield

    def bench():
        # write fully
        for i in range(depth):
            yield from test_case(dut, 16-i, 1, in_r_en=0)
        # read fully
        for i in range(depth):
            yield from test_case(dut, 0, 0, 1)

        # write fully + 1
        for i in range(depth+1):
            yield from test_case(dut, 16-i, 1, in_r_en=0)
        # read fully + 1
        for i in range(depth+1):
            yield from test_case(dut, 0, 0, 1)

    from pathlib import Path
    p = Path(__file__)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_sync_process(bench)

    with open(p.with_suffix('.vcd'), 'w') as f:
        with sim.write_vcd(f):
            sim.run()

    from amaranth.back import verilog
    top = TestFIFO(width=width, depth=depth)
    with open(p.with_suffix('.v'), 'w') as f:
        f.write(
            verilog.convert(
                top,
                ports=[top.in_data, top.out_w_rdy, top.in_w_en, top.out_w_level,
                       top.out_data, top.out_r_rdy, top.in_r_en, top.
                       out_r_level]))
