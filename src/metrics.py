from sdtw_div.numba_ops import *


class SoftDTW:
    def __init__(self, gamma):
        super(SoftDTW, self).__init__()
        self.gamma = gamma

    def __call__(self, x, y):
        return sdtw(x, y, gamma=self.gamma)


class SoftDTWValueAndGrad:
    def __init__(self, gamma):
        super(SoftDTWValueAndGrad, self).__init__()
        self.gamma = gamma

    def __call__(self, x, y):
        return sdtw_value_and_grad(x, y, gamma=self.gamma)


class SharpSoftDTW:
    def __init__(self, gamma):
        super(SharpSoftDTW, self).__init__()
        self.gamma = gamma

    def __call__(self, x, y):
        return sharp_sdtw(x, y, gamma=self.gamma)


class SharpSoftDTWValueAndGrad:
    def __init__(self, gamma):
        super(SharpSoftDTWValueAndGrad, self).__init__()
        self.gamma = gamma

    def __call__(self, x, y):
        return sharp_sdtw_value_and_grad(x, y, gamma=self.gamma)


class SoftDTWDiv:
    def __init__(self, gamma):
        super(SoftDTWDiv, self).__init__()
        self.gamma = gamma

    def __call__(self, x, y):
        return sdtw_div(x, y, gamma=self.gamma)


class SoftDTWDivValueAndGrad:
    def __init__(self, gamma):
        super(SoftDTWDivValueAndGrad, self).__init__()
        self.gamma = gamma

    def __call__(self, x, y):
        return sdtw_div_value_and_grad(x, y, gamma=self.gamma)


class SharpSoftDTWDiv:
    def __init__(self, gamma):
        super(SharpSoftDTWDiv, self).__init__()
        self.gamma = gamma

    def __call__(self, x, y):
        return sharp_sdtw_div(x, y, gamma=self.gamma)


class SharpSoftDTWDivValueAndGrad:
    def __init__(self, gamma):
        super(SharpSoftDTWDivValueAndGrad, self).__init__()
        self.gamma = gamma

    def __call__(self, x, y):
        return sharp_sdtw_div_value_and_grad(x, y, gamma=self.gamma)
