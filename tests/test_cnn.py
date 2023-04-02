import torch
import sdg.viz.splinecam as sc


GLOBAL_DTYPE = torch.float32  # TODO: does reducing 64->32 introduce numerical errors?
GLOBAL_DEVICE = "mps"


def _build_model():
    model = torch.nn.Sequential(
        *[
            torch.nn.Conv2d(3, 6, 5, stride=2, padding=2, bias=False),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.Conv2d(6, 16, 5, stride=2, padding=2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 8 * 8, 120),
            torch.nn.Linear(120, 84),
            torch.nn.Linear(84, 10),
        ]
    )

    model.eval()
    model.type(GLOBAL_DTYPE)
    return model


def _wrap_model(model):
    NN = sc.wrappers.ModelWrapper(
        model,
        input_shape=(3, 32, 32),
        T=torch.randn(3 * 32 * 32, 2),
        device="mps",
        dtype=GLOBAL_DTYPE,
    )
    return NN


def test_viz_splinecam_modelwrap():
    model = _build_model()
    NN = _wrap_model(model)
    assert NN.verify(), "Forward and Affine Equivalence Failed"


def test_viz_cnn_random():
    import matplotlib.pyplot as plt

    model = _build_model()
    NN = _wrap_model(model)
    poly = sc.utils.create_polytope_2d(scale=1, seed=10) + 2
    poly = torch.from_numpy(poly).to(GLOBAL_DEVICE).to(GLOBAL_DTYPE)

    Abw = NN.layers[0].get_weights()[None, ...]
    out_cyc = [poly]

    for current_layer in range(1, len(NN.layers)):
        out_cyc, out_idx = sc.graph.to_next_layer_partition(
            cycles=out_cyc,
            Abw=Abw,
            NN=NN,
            current_layer=current_layer,
            dtype=GLOBAL_DTYPE,
        )

        with torch.no_grad():
            means = sc.utils.get_region_means(
                out_cyc, dims=out_cyc[0].shape[-1], dtype=GLOBAL_DTYPE
            )
            means = NN.layers[:current_layer].forward(means.to(GLOBAL_DEVICE))

            Abw = sc.utils.get_Abw(
                q=NN.layers[current_layer].get_activation_pattern(means),
                Wb=NN.layers[current_layer].get_weights(),
                incoming_Abw=Abw[out_idx],
            )

    sc.plot.plot_partition(
        out_cyc,
        xlims=[poly[:, 0].cpu().min().numpy(), poly[:, 0].cpu().max().numpy()],
        ylims=[poly[:, 1].cpu().min().numpy(), poly[:, 1].cpu().max().numpy()],
    )
    plt.show()
