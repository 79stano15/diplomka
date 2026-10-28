from __future__ import annotations

import math
import time

import torch
from torch import nn


class FastQuantumBlock(nn.Module):
    """Exact differentiable statevector simulation of our circuit.

    Circuit per layer:
        RY(pi * input)
        RY(weight_0)
        RZ(weight_1)
        optional CNOT ring
        RX(weight_2)

    Input rotations are re-uploaded in every layer.
    Wire 0 is the most significant computational-basis bit.
    """

    def __init__(
        self,
        entangle: bool,
        qubits: int = 4,
        depth: int = 2,
    ) -> None:
        super().__init__()

        if qubits < 2 or depth < 1:
            raise ValueError("Expected qubits >= 2 and depth >= 1.")

        self.qubits = qubits
        self.depth = depth
        self.entangle = entangle
        self.dimension = 1 << qubits

        self.weights = nn.Parameter(
            torch.empty(depth, qubits, 3)
        )
        nn.init.uniform_(self.weights, -0.1, 0.1)

        basis = torch.arange(
            self.dimension,
            dtype=torch.long,
        )

        # Map each input basis state through the full CNOT ring.
        destination = basis.clone()

        for wire in range(qubits):
            target = (wire + 1) % qubits

            control_mask = 1 << (qubits - wire - 1)
            target_mask = 1 << (qubits - target - 1)

            control_is_one = (
                destination.bitwise_and(control_mask) != 0
            )

            destination = destination.bitwise_xor(
                control_is_one.to(torch.long) * target_mask
            )

        # output[:, destination[source]] = input[:, source]
        inverse_permutation = torch.argsort(destination)

        z_signs = torch.stack(
            [
                1.0
                - 2.0
                * (
                    (basis >> (qubits - wire - 1)) & 1
                ).to(torch.float64)
                for wire in range(qubits)
            ],
            dim=-1,
        )

        # Derived constants are not stored in checkpoints.
        # The learned state_dict remains {"weights": ...}.
        self.register_buffer(
            "cnot_inverse",
            inverse_permutation,
            persistent=False,
        )
        self.register_buffer(
            "z_signs",
            z_signs,
            persistent=False,
        )

    def _apply_gate(
        self,
        state: torch.Tensor,
        wire: int,
        m00: torch.Tensor,
        m01: torch.Tensor,
        m10: torch.Tensor,
        m11: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]

        paired = state.reshape(
            batch_size,
            1 << wire,
            2,
            1 << (self.qubits - wire - 1),
        )

        amplitude_zero = paired[:, :, 0, :]
        amplitude_one = paired[:, :, 1, :]

        output_zero = (
            m00.reshape(-1, 1, 1) * amplitude_zero
            + m01.reshape(-1, 1, 1) * amplitude_one
        )
        output_one = (
            m10.reshape(-1, 1, 1) * amplitude_zero
            + m11.reshape(-1, 1, 1) * amplitude_one
        )

        return torch.stack(
            (output_zero, output_one),
            dim=2,
        ).reshape(batch_size, self.dimension)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.qubits:
            raise ValueError(
                f"Expected input shape (batch, {self.qubits})."
            )

        # Use double precision internally for the statevector.
        angles = inputs.to(dtype=torch.float64)
        weights = self.weights.to(dtype=torch.float64)

        state = torch.zeros(
            (inputs.shape[0], self.dimension),
            dtype=torch.complex128,
            device=inputs.device,
        )
        state[:, 0] = 1.0

        for layer in range(self.depth):
            for wire in range(self.qubits):
                # Consecutive rotations about the same axis:
                # RY(w) @ RY(x) = RY(x + w).
                angle = (
                    math.pi * angles[:, wire]
                    + weights[layer, wire, 0]
                )

                cosine = torch.cos(angle / 2.0)
                sine = torch.sin(angle / 2.0)

                phase_minus = torch.exp(
                    -0.5j * weights[layer, wire, 1]
                )
                phase_plus = torch.exp(
                    0.5j * weights[layer, wire, 1]
                )

                # Combined matrix RZ(weight_1) @ RY(angle).
                state = self._apply_gate(
                    state,
                    wire,
                    phase_minus * cosine,
                    -phase_minus * sine,
                    phase_plus * sine,
                    phase_plus * cosine,
                )

            if self.entangle:
                state = state.index_select(
                    dim=1,
                    index=self.cnot_inverse,
                )

            for wire in range(self.qubits):
                angle = weights[layer, wire, 2]
                cosine = torch.cos(angle / 2.0)
                off_diagonal = -1j * torch.sin(angle / 2.0)

                state = self._apply_gate(
                    state,
                    wire,
                    cosine,
                    off_diagonal,
                    off_diagonal,
                    cosine,
                )

        probabilities = (
            state.real.square() + state.imag.square()
        )

        expectations = probabilities @ self.z_signs

        return expectations.to(dtype=inputs.dtype)


def verify_against_pennylane() -> None:
    # Keep the original QuantumBlock in the comparison script.
    from compare_quantum_classical import QuantumBlock

    generator = torch.Generator().manual_seed(2026)

    for entangle in (False, True):
        reference = QuantumBlock(entangle=entangle)
        fast = FastQuantumBlock(entangle=entangle)

        # Verify beyond the small initialization interval.
        test_weights = (
            2.0
            * math.pi
            * torch.rand(
                reference.weights.shape,
                generator=generator,
            )
            - math.pi
        )

        with torch.no_grad():
            reference.weights.copy_(test_weights)

        fast.load_state_dict(reference.state_dict())

        for batch_size in (1, 17):
            inputs = (
                2.0
                * torch.rand(
                    (batch_size, 4),
                    generator=generator,
                )
                - 1.0
            )

            reference_inputs = (
                inputs.clone().requires_grad_(True)
            )
            fast_inputs = inputs.clone().requires_grad_(True)

            reference_outputs = reference(reference_inputs)
            fast_outputs = fast(fast_inputs)

            torch.testing.assert_close(
                fast_outputs,
                reference_outputs,
                rtol=1e-5,
                atol=3e-6,
            )

            coefficients = torch.randn(
                (4,),
                generator=generator,
            )

            def objective(outputs: torch.Tensor) -> torch.Tensor:
                logits = outputs @ coefficients

                return (
                    -torch.log_softmax(logits, dim=0)[0]
                    + 0.1 * outputs.square().mean()
                )

            reference_gradients = torch.autograd.grad(
                objective(reference_outputs),
                (reference_inputs, reference.weights),
            )
            fast_gradients = torch.autograd.grad(
                objective(fast_outputs),
                (fast_inputs, fast.weights),
            )

            for actual, expected in zip(
                fast_gradients,
                reference_gradients,
            ):
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-3,
                    atol=3e-5,
                )

            output_error = (
                fast_outputs - reference_outputs
            ).abs().max().item()

            input_gradient_error = (
                fast_gradients[0] - reference_gradients[0]
            ).abs().max().item()

            weight_gradient_error = (
                fast_gradients[1] - reference_gradients[1]
            ).abs().max().item()

            print(
                f"OK: entangle={entangle}, batch={batch_size}; "
                f"output error={output_error:.3e}, "
                f"input-grad error={input_gradient_error:.3e}, "
                f"weight-grad error={weight_gradient_error:.3e}"
            )


def benchmark() -> None:
    from compare_quantum_classical import QuantumBlock

    reference = QuantumBlock(entangle=True)
    fast = FastQuantumBlock(entangle=True)
    fast.load_state_dict(reference.state_dict())

    generator = torch.Generator().manual_seed(321)
    inputs = torch.randn(
        (32, 4),
        generator=generator,
    ).requires_grad_(True)

    repetitions = 30

    def measure(
        model: nn.Module,
        backward: bool,
    ) -> float:
        def step() -> None:
            with torch.set_grad_enabled(backward):
                outputs = model(inputs)

                if backward:
                    torch.autograd.grad(
                        outputs.square().mean(),
                        (inputs, model.weights),
                    )

        for _ in range(5):
            step()

        started = time.perf_counter()

        for _ in range(repetitions):
            step()

        return (
            time.perf_counter() - started
        ) / repetitions

    print("\nQuantum-block benchmark: batch=32, CPU threads=1")

    for backward in (False, True):
        reference_time = measure(reference, backward)
        fast_time = measure(fast, backward)

        label = (
            "forward + backward"
            if backward
            else "forward only"
        )

        print(
            f"{label}: "
            f"PennyLane={1000 * reference_time:.3f} ms, "
            f"Torch={1000 * fast_time:.3f} ms, "
            f"ratio={reference_time / fast_time:.2f}x"
        )


def main() -> None:
    torch.set_num_threads(1)
    verify_against_pennylane()
    benchmark()


if __name__ == "__main__":
    main()