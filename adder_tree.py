from amaranth import *


def is_power_of_two(x):
    return (x & (x - 1)) == 0


class AdderTree(Elaboratable):
    def __init__(self, acc_bits, fan_in, signed=True):
        self.acc_bits = acc_bits
        self.fan_in = fan_in
        self.signed = signed
        assert is_power_of_two(fan_in)
        assert fan_in >= 2

        self.in_data = Array([
            Signal(Shape(acc_bits, signed=signed),
                   name=f'in_data_{fan_in}_{i}')
            for i in range(fan_in)])
        self.in_ovf = Array([Signal(1, name=f'in_ovf_{fan_in}_{i}')
                             for i in range(fan_in)])
        self.in_valid = Array([Signal(1, name=f'in_valid_{fan_in}_{i}')
                               for i in range(fan_in)])
        self.out_d = Signal(Shape(acc_bits, signed=signed))
        self.out_ovf = Signal(1)
        self.out_valid = Signal(1)

        # TODO
        if self.fan_in > 2:
            self.tree_l = AdderTree(acc_bits, fan_in // 2, signed)
            self.tree_r = AdderTree(acc_bits, fan_in // 2, signed)

    def elaborate(self, platform):
        m = Module()

        sum_result = Signal(Shape(self.acc_bits + 1, signed=self.signed))
        acc_min = -(1 << (self.acc_bits - 1))
        acc_max = (1 << (self.acc_bits - 1)) - 1

        if self.fan_in == 2:
            m.d.comb += [
                sum_result.eq(self.in_data[0] + self.in_data[1]),
                self.out_d.eq(sum_result[:-1]),
                self.out_valid.eq(self.in_valid[0] & self.in_valid[1]),
            ]
            with m.If((sum_result < acc_min) | (sum_result > acc_max)):
                m.d.comb += [
                    self.out_ovf.eq(1),
                ]
        else:
            m.submodules.tree_l = self.tree_l
            m.submodules.tree_r = self.tree_r

            for i in range(self.fan_in // 2):
                m.d.comb += [
                    self.tree_l.in_data[i].eq(self.in_data[i]),
                    self.tree_l.in_ovf[i].eq(self.in_ovf[i]),
                    self.tree_l.in_valid[i].eq(self.in_valid[i]),
                    self.tree_r.in_data[i].eq(self.in_data[i + self.fan_in // 2]),
                    self.tree_r.in_ovf[i].eq(self.in_ovf[i + self.fan_in // 2]),
                    self.tree_r.in_valid[i].eq(self.in_valid[i + self.fan_in // 2]),
                ]

            m.d.comb += [
                sum_result.eq(self.tree_l.out_d + self.tree_r.out_d),
                self.out_d.eq(sum_result[:-1]),
                self.out_valid.eq(self.tree_l.out_valid & self.tree_r.out_valid),
            ]

            with m.If((sum_result < acc_min) | (sum_result > acc_max)):
                m.d.comb += [
                    self.out_ovf.eq(1),
                ]
            with m.Else():
                m.d.comb += [
                    self.out_ovf.eq(self.tree_l.out_ovf | self.tree_r.out_ovf),
                ]
        
        return m


if __name__ == '__main__':
    acc_bits = 8
    fan_in = 4
    signed = True
    dut = AdderTree(acc_bits=acc_bits, fan_in=fan_in, signed=signed)

    import numpy as np

    from amaranth.sim import Delay, Settle, Simulator

    np.random.seed(42)

    def test_case(dut, in_data, in_ovf, in_valid):
        for i, d in enumerate(in_data):
            yield dut.in_data[i].eq(int(d))
        for i, d in enumerate(in_ovf):
            yield dut.in_ovf[i].eq(bool(d))
        for i, d in enumerate(in_valid):
            yield dut.in_valid[i].eq(bool(d))
        yield Settle()
        yield Delay(1e-6)
        hw_sum = yield dut.out_d
        hw_ovf = yield dut.out_ovf
        print(f'hw_sum: {hw_sum}, hw_ovf: {hw_ovf}')
        if hw_ovf:
            assert ((hw_sum - sum(in_data)) % (2 ** acc_bits) == 0)
        else:
            assert (hw_sum == sum(in_data))

    def bench():
        for _ in range(512):
            yield from test_case(dut,
                                 in_data=np.random.randint(
                                     low=-2**(acc_bits-2),
                                     high=2**(acc_bits-2), size=fan_in),
                                 in_ovf=np.random.randint(
                                     low=0, high=1, size=fan_in),
                                 in_valid=np.random.randint(
                                     low=0,  high=2, size=fan_in)
                                 )

    from pathlib import Path
    p = Path(__file__)

    sim = Simulator(dut)
    sim.add_process(bench)

    with open(p.with_suffix('.vcd'), 'w') as f:
        with sim.write_vcd(f):
            sim.run()

    from amaranth.back import verilog
    top = AdderTree(acc_bits=acc_bits, fan_in=fan_in, signed=signed)
    with open(p.with_suffix('.v'), 'w') as f:
        f.write(
            verilog.convert(
                top,
                ports=[*top.in_data, *top.in_ovf, *top.in_valid,
                       top.out_d, top.out_ovf, top.out_valid]))
