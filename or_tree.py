from amaranth import *


def is_power_of_two(x):
    return (x & (x - 1)) == 0


class ORTree(Elaboratable):
    def __init__(self, num_bits, fan_in):
        self.num_bits = num_bits
        self.fan_in = fan_in

        assert is_power_of_two(fan_in)

        self.in_arr = Array(
            [Signal(num_bits, name=f"in_{i}_{fan_in}") for i in range(fan_in)]
        )
        self.out_d = Signal(num_bits, name=f"out_{fan_in}")

        self.tree_l = None
        self.tree_r = None
        if self.fan_in > 2:
            self.tree_l = ORTree(num_bits, fan_in // 2)
            self.tree_r = ORTree(num_bits, fan_in // 2)

    def elaborate(self, platform):
        m = Module()

        if self.fan_in > 2:
            m.submodules.tree_l = tree_l = self.tree_l
            m.submodules.tree_r = tree_r = self.tree_r

            half = self.fan_in // 2
            for i in range(half):
                m.d.comb += [
                    tree_l.in_arr[i].eq(self.in_arr[i]),
                    tree_r.in_arr[i].eq(self.in_arr[i + half]),
                    self.out_d.eq(tree_l.out_d | tree_r.out_d),
                ]
        else:
            m.d.comb += [
                self.out_d.eq(self.in_arr[0] | self.in_arr[1]),
            ]

        return m


if __name__ == "__main__":
    num_bits = 32
    fan_in = 8
    dut = ORTree(num_bits=num_bits, fan_in=fan_in)

    from amaranth.sim import Simulator
    from amaranth.sim import Settle, Delay
    import numpy as np
    from functools import reduce

    np.random.seed(42)

    def test_case(dut, in_arr):
        for i in range(len(in_arr)):
            yield dut.in_arr[i].eq(int(in_arr[i]))
        yield Settle()
        yield Delay(1e-6)
        out_val = yield dut.out_d
        assert out_val == reduce(lambda acc, cur: acc | cur, in_arr, 0)

    def bench():
        num_test = 256
        for _ in range(num_test):
            yield from test_case(
                dut, np.random.randint(low=0, high=2**num_bits, size=[fan_in])
            )

    from pathlib import Path

    p = Path(__file__)

    sim = Simulator(dut)
    sim.add_process(bench)

    with open(p.with_suffix(".vcd"), "w") as f:
        with sim.write_vcd(f):
            sim.run()

    from amaranth.back import verilog

    top = ORTree(num_bits=num_bits, fan_in=fan_in)
    with open(p.with_suffix(".v"), "w") as f:
        f.write(verilog.convert(top, ports=[*top.in_arr, top.out_d]))
