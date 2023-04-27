from amaranth import *


class MAC(Elaboratable):
    def __init__(self, num_bits, acc_bits, signed=True):
        self.num_bits = num_bits
        self.acc_bits = acc_bits
        self.signed = signed

        assert acc_bits >= num_bits * 2

        self.in_a = Signal(Shape(num_bits, signed=signed))
        self.in_a_valid = Signal(1)
        self.in_b = Signal(Shape(num_bits, signed=signed))
        self.in_b_valid = Signal(1)

        self.in_rst = Signal(1, reset_less=True)

        self.out_d = Signal(Shape(acc_bits, signed=signed))
        self.out_d_valid = Signal(1)
        self.out_ovf = Signal(1)

    def elaborate(self, platform):
        m = Module()

        mul = Signal(self.out_d.shape())
        sum = Signal(self.out_d.shape())
        m.d.comb += [
            mul.eq(self.in_a * self.in_b),
            sum.eq(mul + self.out_d)
        ]

        with m.If(self.in_a_valid & self.in_b_valid):
            aneg = self.out_d < 0
            bneg = mul < 0
            dneg = sum < 0

            m.d.sync += [
                self.out_d.eq(sum),
                self.out_ovf.eq(self.out_ovf | ((aneg != dneg) & (bneg != dneg))),
                self.out_d_valid.eq(1),
            ]

        return m


if __name__ == '__main__':
    num_bits = 4
    acc_bits = 8
    signed = True
    dut = MAC(num_bits=num_bits, acc_bits=acc_bits, signed=signed)
    dut = ResetInserter(dut.in_rst)(dut)  # reset signal for simulation

    from amaranth.sim import Simulator
    import numpy as np

    np.random.seed(10)

    def test_case(dut, a, a_valid, b, b_valid, rst):
        yield dut.in_a.eq(int(a))
        yield dut.in_a_valid.eq(a_valid)
        yield dut.in_b.eq(int(b))
        yield dut.in_b_valid.eq(b_valid)
        yield dut.in_rst.eq(rst)
        yield

    def bench():
        # valid input should affect
        yield from test_case(dut, 1, 1, 1, 1, 0)
        # invalid input should not affect
        yield from test_case(dut, 1, 0, 1, 0, 0)
        # reset
        yield from test_case(dut, 1, 0, 1, 0, 1)

        for i in range(5):
            for j in range(2 ** num_bits):
                rdm = np.random.randint(low=0, high=2 ** num_bits, size=[2])
                if signed:
                    rdm -= 2 ** (num_bits-1)
                yield from test_case(dut, rdm[0], 1, rdm[1], 1, 0)
            # reset
            yield from test_case(dut, i, 1, j, 1, 1)

    from pathlib import Path
    p = Path(__file__)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_sync_process(bench)

    with open(p.with_suffix('.vcd'), 'w') as f:
        with sim.write_vcd(f):
            sim.run()

    from amaranth.back import verilog
    top = MAC(num_bits=num_bits, acc_bits=acc_bits, signed=signed)
    with open(p.with_suffix('.v'), 'w') as f:
        f.write(verilog.convert(
            top,
            # NOTE `rst` is automatically generated
            ports=[top.in_a, top.in_a_valid, top.in_b, top.in_b_valid,
                   top.out_d, top.out_d_valid, top.out_ovf]))
