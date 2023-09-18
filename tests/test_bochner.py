import torch

from src.kernels.bochner import BochnerKernel
from src.kernels.gaussian import GaussianKernel, MultiDimGaussianKernel

class TestBochnerKernel:
    @staticmethod
    def slow_bochner_kernel(
        X: torch.Tensor,
        Y: torch.Tensor,
        Q: torch.Tensor,
        b: torch.Tensor
    ):
        result = torch.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                difference = x - y
                cos_term = torch.cos(torch.inner(difference, b))
                exp_term = torch.exp(
                    -torch.norm(
                        torch.matmul(Q.T, difference)
                    ) ** 2 / 2
                )
                result[i, j] = cos_term * exp_term
        return result
    
    @staticmethod
    def test_evaluate_analytically():
        DIM = 5
        N_X = 100
        N_Y = 200

        bochner = BochnerKernel(DIM, device='cpu')
        X = torch.rand((N_X, DIM))
        Y = torch.rand((N_Y, DIM))

        Q = bochner.linear.weight
        b = bochner.linear.bias

        expected = TestBochnerKernel.slow_bochner_kernel(X, Y, Q, b)
        actual = bochner.evaluate_analytically(X, Y)

        assert torch.allclose(expected, actual)

    @staticmethod
    def test_call():
        DIM = 5
        N_X =100
        N_Y = 200

        bochner = BochnerKernel(DIM, device='cpu')
        X = torch.rand((N_X, DIM))
        Y = torch.rand((N_Y, DIM))

        expected = bochner.evaluate_analytically(X, Y)
        actual = bochner(X, Y)

        assert torch.allclose(expected, actual)

    @staticmethod
    def test_emulate_gaussian():
        DIM = 5
        N_X = 100
        N_Y = 200

        lengthscale = torch.rand(1)

        gaussian = GaussianKernel(lengthscale)
        bochner = BochnerKernel.from_gaussian_kernel(DIM, lengthscale, device='cpu')

        X = torch.rand((N_X, DIM))
        Y = torch.rand((N_Y, DIM))

        expected = gaussian(X, Y)
        actual = bochner.evaluate_analytically(X, Y)
        
        assert torch.allclose(expected, actual)

    @staticmethod
    def test_emulate_multidim_gaussian():
        DIM = 5
        N_X = 100
        N_Y = 200

        lengthscales = torch.Tensor([1.,2.,3.,4.,5.])

        gaussian = MultiDimGaussianKernel(lengthscales)
        bochner = BochnerKernel.from_multidim_gaussian_kernel(DIM, lengthscales, device='cpu')

        X = torch.rand((N_X, DIM))
        Y = torch.rand((N_Y, DIM))

        expected = gaussian(X, Y)
        actual = bochner.evaluate_analytically(X, Y)

        assert torch.allclose(expected, actual)

    @staticmethod
    def test_differences():
        DIM = 5
        N_X = 100
        N_Y = 200

        X = torch.rand((N_X, DIM))
        Y = torch.rand((N_Y, DIM))

        fast = (X.unsqueeze(1) - Y.unsqueeze(0))
        slow = torch.zeros((N_X, N_Y, DIM))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                slow[i, j] = x - y
        # print(f'Max difference was {(fast-slow).max().item():.4e}')
        assert torch.allclose(fast, slow)

    @staticmethod
    def test_einsum():
        DIM = 5
        N_X = 10
        N_Y = 20
        N_SAMPLES = 1000

        X = torch.rand((N_X, DIM))
        Y = torch.rand((N_Y, DIM))
        samples = torch.rand((N_SAMPLES, DIM))

        fast = torch.einsum(
            'xyi,si->xys',
            (X.unsqueeze(1) - Y.unsqueeze(0)),
            samples
        )
        slow = torch.zeros((N_X, N_Y, N_SAMPLES))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                for k, s in enumerate(samples):
                    slow[i, j, k] = torch.inner(x-y, s)
        assert torch.allclose(fast, slow, atol=1e-5)

    

    @staticmethod
    def test_sampled_version():
        DIM = 2
        N_X = 100
        N_Y = 200

        # Initialising a bochner kernel automatically randomizes parameters
        bochner = BochnerKernel(DIM, device='cpu')

        X = torch.rand((N_X, DIM))
        Y = torch.rand((N_Y, DIM))

        expected = bochner.evaluate_analytically(X, Y)

        samples = bochner.sample(10_000).detach()
        actual = bochner.evaluate_from_samples(X, Y, samples)

        print("")
        print(expected)
        print(actual)

        # assert torch.allclose(expected, actual)

